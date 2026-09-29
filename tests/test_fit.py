import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from graphspec.infra import compute
from graphspec.infra.files import read_json
from tuning import api
from tuning.data import Bank
from tuning.run import flags, run
from tuning.statement import names, state
from tuning.training import fit

LOG = logging.getLogger(__name__)


def arguments(bank, output, statement):
    return argparse.Namespace(
        bank=str(bank),
        output=str(output),
        device=statement["device"],
        seed=statement["seed"],
        seconds=statement["seconds"],
        preparation_seconds=statement["preparation_seconds"],
        workers=statement["workers"],
        variant_gb=statement["variant_cache_bytes"] >> 30,
        batch_sequences=8,
        continuous=False,
        trials=None,
        selector=None,
        random_trials=None,
        random_params=None,
        mix=None,
        objective=None,
        bounds=None,
    )


def test_synthetic_fit_completes_on_cpu(bank, packed, tmp_path, statement):
    opened, cfg = Bank(bank, device="cpu"), packed["snapshot"]
    blocks = opened.blocks(0, 4, cfg)
    started = time.perf_counter()
    result = fit(bank, tmp_path, cfg, blocks, 16, started + statement["seconds"], statement)
    seconds = time.perf_counter() - started
    LOG.info("cpu fit exit seconds=%.3f timings=%s", seconds, result["timings"])
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and saved["peak_cuda_bytes"] == 0
    assert saved["statement"] == statement
    assert 0 < saved["optimizer_updates"] <= 20
    assert saved["train_unique_rows"] == len(opened.split["train_indices"])
    weights = torch.load(tmp_path / "checkpoint.pt", weights_only=False)["state_dict"]
    assert all(value.device.type == "cpu" for value in weights.values())


def test_fit_on_an_absent_gpu_stops_naming_the_device(bank, tmp_path, monkeypatch, statement):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="device cuda:0 was requested and no GPU is present"):
        fit(bank, tmp_path, {}, [], 8, time.perf_counter() + 30, dict(statement, device="cuda:0"))
    assert not list(tmp_path.iterdir())


def test_dense_algebra_on_an_absent_gpu_stops_naming_the_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    large = np.eye(compute.GPU_FROM)
    for function in (compute.inverse, compute.eigh, lambda values, device: compute.top(values, 1, device)):
        with pytest.raises(RuntimeError, match="device cuda was requested and no GPU is present"):
            function(large, "cuda")
    assert compute.device_for(compute.GPU_FROM - 1, "cuda") == "cpu"
    assert compute.device_for(compute.GPU_FROM, "cpu") == "cpu"


def test_a_bank_opened_without_a_device_cannot_derive(bank):
    with pytest.raises(ValueError, match="opened without a device"):
        Bank(bank).limits()


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
        fit(tmp_path / "bank", tmp_path, {}, [], 8, time.perf_counter() + 30, given)
    with pytest.raises(KeyError, match="the statement lacks " + name):
        api.fit.__wrapped__(tmp_path / "bank", {}, tmp_path / "fit", given)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", names())
def test_search_with_a_field_absent_stops_naming_the_field(bank, tmp_path, statement, name):
    args = arguments(bank, tmp_path / "run", statement)
    given = flags(args)
    assert state(given).mapping() == statement
    del given[name]
    with pytest.raises(KeyError, match="the statement lacks " + name):
        state(given)
    flag = dict(variant_cache_bytes="variant_gb").get(name, name)
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


def test_fit_rejects_a_configuration_whose_seed_is_not_the_statements(tmp_path, statement):
    with pytest.raises(ValueError, match="the seed of the configuration is 6"):
        fit(tmp_path / "bank", tmp_path, dict(seed=6), [], 8, time.perf_counter() + 30, statement)
