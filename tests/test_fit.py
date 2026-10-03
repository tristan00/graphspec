import argparse
import dataclasses
import logging
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from support import SEED

from graphspec.infra import compute
from graphspec.infra.arrays import RESIDENT
from graphspec.infra.files import read_json
from tuning import api
from tuning.bank.bank import Bank, Deriver
from tuning.fits.training import fit
from tuning.searches.run import flags, run
from tuning.statement import Statement, names, paired, state

LOG = logging.getLogger(__name__)


def arguments(bank, output, statement):
    return argparse.Namespace(
        bank=str(bank),
        output=str(output),
        device=statement["device"],
        seed=statement["seed"],
        derivation_seed=statement["derivation_seed"],
        seconds=statement["seconds"],
        preparation_seconds=statement["preparation_seconds"],
        workers=statement["workers"],
        variant_gb=statement["variant_cache_bytes"] >> 30,
        batch_sequences=statement["batch_sequences"],
        trained=statement["trained"],
        scored=statement["scored"],
        sees=["%s=%s" % (name, ",".join(rules)) for name, rules in statement["sees"].items()],
        continuous=False,
        trials=None,
        selector="random",
        random_trials=10,
        random_params=[],
        mix=["random=100"],
        objective=None,
        bounds=None,
        loss=["%s=%s" % pair for pair in statement["loss"].items()],
        measures=statement["measures"],
        alike=statement["alike"],
    )


def test_synthetic_fit_completes_on_cpu(bank, packed, tmp_path, statement):
    opened, cfg = Deriver(bank, "cpu", SEED, RESIDENT), packed["snapshot"]
    blocks = opened.blocks(0, 4, cfg)
    started = time.perf_counter()
    result = fit(bank, tmp_path, cfg, blocks, 16, started + statement["seconds"], state(statement))
    seconds = time.perf_counter() - started
    LOG.info("cpu fit exit seconds=%.3f timings=%s", seconds, result["timings"])
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and saved["peak_cuda_bytes"] == 0
    assert saved["statement"] == statement
    assert 0 < saved["optimizer_updates"] <= 20
    assert saved["train_unique_rows"] == len(opened.members(cfg, 0))
    weights = torch.load(tmp_path / "checkpoint.pt", weights_only=False)["state_dict"]
    assert all(value.device.type == "cpu" for value in weights.values())


def test_fit_on_an_absent_gpu_stops_naming_the_device(bank, tmp_path, monkeypatch, statement):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    stated = state(dict(statement, device="cuda:0"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="device cuda:0 was requested and no GPU is present"):
        fit(bank, tmp_path, {}, [], 8, time.perf_counter() + 30, stated)
    assert not list(tmp_path.iterdir())


def test_a_device_whose_index_is_absent_stops_naming_the_device_as_stated(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="device cuda:9 was requested and 2 GPUs are present"):
        compute.require("cuda:9")
    found = compute.require("cuda:1")
    assert (found.type, found.index) == ("cuda", 1)
    assert compute.require("cpu").type == "cpu"
    with pytest.raises(ValueError, match="device meta is neither cpu nor cuda"):
        compute.require("meta")


def test_dense_algebra_on_an_absent_gpu_stops_naming_the_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    large = np.eye(compute.GPU_FROM)
    for function in (compute.inverse, compute.eigh, lambda values, device: compute.top(values, 1, device)):
        with pytest.raises(RuntimeError, match="device cuda was requested and no GPU is present"):
            function(large, "cuda")
    assert compute.device_for(compute.GPU_FROM - 1, "cuda") == "cpu"
    assert compute.device_for(compute.GPU_FROM, "cpu") == "cpu"


def test_a_bank_opened_to_read_cannot_derive_and_deriving_states_its_device(bank):
    opened = Bank(bank)
    assert opened.registry.fingerprint() == opened.manifest["spec_sha256"]
    for name in ("limits", "identity", "variant", "stamp", "arrays", "resident", "sizes", "segmented"):
        assert not hasattr(opened, name) and hasattr(Deriver, name)
    with pytest.raises(TypeError, match="device"):
        Deriver(bank)
    with pytest.raises(TypeError, match="seed"):
        Deriver(bank, "cpu", resident=RESIDENT)
    for device in (None, "gpu", 0):
        with pytest.raises(ValueError, match="the device of graph limits is %r" % (device,)):
            Deriver(bank, device, SEED, RESIDENT)


def test_the_key_of_a_variant_holds_the_device(bank):
    from support import configuration

    from conftest import SPLITS

    cfg = configuration(Bank(bank).registry, dict(view="snapshot"), {"time/history": 4}, 5, SPLITS)
    central, graphic = Deriver(bank, "cpu", SEED, RESIDENT), Deriver(bank, "cuda", SEED, RESIDENT)
    left, right = central.identity(cfg), graphic.identity(cfg)
    assert {name for name in left if left[name] != right[name]} == {"device"}
    assert sorted(left) == [
        "added",
        "device",
        "entries",
        "groups",
        "pipeline",
        "schema",
        "seeds",
        "settings",
        "source",
        "splits",
        "versions",
    ]
    assert central.variant(cfg)[0].parent == central.cache / "variants"
    assert central.variant(cfg)[0] != graphic.variant(cfg)[0]
    assert central.variant(cfg)[0] == Deriver(bank, "cpu", SEED, RESIDENT).variant(cfg)[0]
    with pytest.raises(ValueError, match="the device of graph limits is None"):
        Deriver(bank, None, SEED, RESIDENT)
    assert Bank(bank).joining(cfg) is False


def test_the_device_is_not_carried_by_the_environment():
    name = "GRAPHSPEC_" + "DEVICE"
    root = Path(__file__).parent.parent
    found = [
        str(path.relative_to(root))
        for folder in ("graphspec", "tuning")
        for path in sorted((root / folder).rglob("*"))
        if path.is_file() and path.suffix != ".pyc" and name.encode() in path.read_bytes()
    ]
    assert found == []


@pytest.mark.parametrize("name", names())
def test_fit_with_a_field_absent_stops_naming_the_field(tmp_path, statement, name):
    given = {key: value for key, value in statement.items() if key != name}
    with pytest.raises(KeyError, match="the statement lacks " + name):
        api.fit.__wrapped__(tmp_path / "bank", {}, tmp_path / "fit", given)
    with pytest.raises(KeyError, match="the statement lacks " + name):
        Statement(**given)
    with pytest.raises(TypeError, match="a statement built by tuning.statement.state is required"):
        fit(tmp_path / "bank", tmp_path, {}, [], 8, time.perf_counter() + 30, given)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "name,value",
    [
        ("version", 0),
        ("device", "gpu"),
        ("device", "meta"),
        ("device", "cuda:9"),
        ("device", "cuda:-1"),
        ("device", 0),
        ("device", None),
        ("seed", -1),
        ("seed", 2**32),
        ("seed", 1.5),
        ("seed", True),
        ("derivation_seed", -1),
        ("derivation_seed", 2**64),
        ("derivation_seed", 1.5),
        ("derivation_seed", True),
        ("workers", 0),
        ("workers", os.cpu_count() + 1),
        ("workers", 1.0),
        ("seconds", 0),
        ("seconds", 2.5),
        ("preparation_seconds", 0),
        ("variant_cache_bytes", 0),
        ("variant_cache_bytes", -1),
        ("batch_sequences", 0),
        ("batch_sequences", None),
        ("trained", []),
        ("trained", "train"),
        ("trained", ["train", "train"]),
        ("scored", []),
        ("scored", [""]),
        ("sees", ["train", "validation"]),
        ("sees", dict(train=["unregistered"], validation=[])),
        ("sees", dict(train="without_targets", validation=[])),
        ("loss", ""),
        ("loss", None),
        ("loss", "squared"),
        ("loss", {}),
        ("loss", dict(number="")),
        ("measures", []),
        ("measures", "mse"),
        ("measures", ["mse", "mse"]),
        ("alike", -0.1),
        ("alike", 1.5),
        ("alike", True),
        ("alike", None),
    ],
)
def test_a_statement_with_a_value_outside_its_meaning_stops_naming_it(monkeypatch, statement, name, value):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    message = "the statement's %s is %r and must be" % (name, value)
    for build in (state, lambda mapping: Statement(**mapping)):
        with pytest.raises(ValueError) as raised:
            build(dict(statement, **{name: value}))
        assert message in str(raised.value)
    with pytest.raises(ValueError) as raised:
        dataclasses.replace(state(statement), **{name: value})
    assert message in str(raised.value)
    with pytest.raises(ValueError) as raised:
        state(statement).replace(**{name: value})
    assert message in str(raised.value)


def test_a_search_states_the_loss_of_every_kind_of_target_as_a_pair_of_kind_and_loss():
    assert paired(["number=squared", "class=entropy"]) == {"number": "squared", "class": "entropy"}
    for wrong in (["number"], ["number=squared", "number=balanced"], ["=squared"], ["number="]):
        with pytest.raises(ValueError, match="stated once for each kind as kind=loss; found"):
            paired(wrong)


def test_a_statement_is_built_only_through_its_checks(monkeypatch, statement):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(TypeError):
        Statement(*statement.values())
    with pytest.raises(TypeError, match="read from a mapping"):
        state(list(statement.items()))
    held = state(dict(statement, device="cuda:1", seed=2**32 - 1, workers=os.cpu_count()))
    assert (held.device, held.seed, held.workers) == ("cuda:1", 2**32 - 1, os.cpu_count())
    with pytest.raises(dataclasses.FrozenInstanceError):
        held.seed = 3


@pytest.mark.parametrize("name", names())
def test_search_with_a_field_absent_stops_naming_the_field(bank, tmp_path, statement, name):
    args = arguments(bank, tmp_path / "run", statement)
    given = flags(args)
    assert state(given).mapping() == statement
    del given[name]
    with pytest.raises(KeyError, match="the statement lacks " + name):
        state(given)
    flag = dict(variant_cache_bytes="variant_gb")[name] if name == "variant_cache_bytes" else name
    if hasattr(args, flag):
        setattr(args, flag, None)
        with pytest.raises(KeyError, match="the statement lacks " + name):
            run.__wrapped__(args)
        assert not (tmp_path / "run").exists()


def test_statement_holds_every_field_and_no_other(statement):
    assert set(names()) == set(statement)
    assert state(statement).mapping() == statement
    assert state(statement).replace(seed=9).seed == 9
    with pytest.raises(KeyError, match="does not know: colour"):
        state(dict(statement, colour="red"))
    with pytest.raises(ValueError, match="the statement's version is 0"):
        state(dict(statement, version=0))
    with pytest.raises(ValueError, match="the statement's device is 'gpu'"):
        state(dict(statement, device="gpu"))


def test_a_statement_names_every_split_once_and_says_what_each_may_see(statement):
    held = state(statement)
    assert held.splits() == dict(trained=["train"], scored=["validation"], sees=dict(train=[], validation=[]))
    with pytest.raises(ValueError, match="the statement trains on and scores on the same splits: train"):
        state(dict(statement, scored=["train"], sees=dict(train=[])))
    with pytest.raises(ValueError, match="the statement's sees names the splits train and must name"):
        state(dict(statement, sees=dict(train=[])))
    with pytest.raises(ValueError, match="the statement's sees names the splits other, train, validation"):
        state(dict(statement, sees=dict(train=[], validation=[], other=[])))
    seen = state(dict(statement, sees=dict(train=["without_targets", "without_scored"], validation=[])))
    assert seen.sees["train"] == ["without_targets", "without_scored"]


def test_fit_rejects_a_configuration_whose_splits_are_not_the_statements(tmp_path, statement):
    from conftest import SPLITS

    given = dict(seed=statement["seed"])
    with pytest.raises(KeyError, match="the configuration states no splits"):
        fit(tmp_path / "bank", tmp_path, given, [], 8, time.perf_counter() + 30, state(statement))
    other = dict(SPLITS, sees=dict(train=["without_targets"], validation=[]))
    with pytest.raises(ValueError, match="the splits of the configuration are"):
        fit(
            tmp_path / "bank",
            tmp_path,
            dict(given, splits=other),
            [],
            8,
            time.perf_counter() + 30,
            state(statement),
        )
    assert not list(tmp_path.iterdir())


def test_fit_rejects_a_configuration_whose_seed_is_not_the_statements(tmp_path, statement):
    with pytest.raises(ValueError, match="the seed of the configuration is 6"):
        fit(tmp_path / "bank", tmp_path, dict(seed=6), [], 8, time.perf_counter() + 30, state(statement))
