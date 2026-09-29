import logging
import time

import pytest
import torch

from graphspec.infra import compute
from graphspec.infra.files import read_json
from tuning.data import Bank
from tuning.training import fit

LOG = logging.getLogger(__name__)


def test_synthetic_fit_completes_on_cpu(bank, packed, tmp_path):
    opened, cfg = Bank(bank), packed["snapshot"]
    blocks = opened.blocks(0, 4, cfg)
    started = time.perf_counter()
    result = fit(bank, tmp_path, cfg, blocks, 8, started + 30, 1, "cpu")
    seconds = time.perf_counter() - started
    LOG.info("cpu fit exit seconds=%.3f timings=%s", seconds, result["timings"])
    saved = read_json(tmp_path / "result.json")
    assert saved["status"] == "complete" and saved["peak_cuda_bytes"] == 0
    assert 0 < saved["optimizer_updates"] <= 20
    assert saved["train_unique_rows"] == len(opened.split["train_indices"])
    state = torch.load(tmp_path / "checkpoint.pt", weights_only=False)["state_dict"]
    assert all(value.device.type == "cpu" for value in state.values())


def test_fit_on_an_absent_gpu_stops_naming_the_device(bank, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="device cuda:0 was requested and no GPU is present"):
        fit(bank, tmp_path, {}, [], 8, time.perf_counter() + 30, 1, "cuda:0")
    assert not (tmp_path / "result.json").exists()


def test_dense_algebra_on_an_absent_gpu_stops_naming_the_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="device cuda was requested and no GPU is present"):
        compute.state("cuda")
    monkeypatch.setenv(compute.SETTING, "cuda")
    with pytest.raises(RuntimeError, match="device cuda was requested and no GPU is present"):
        compute.device_for(compute.GPU_FROM)
    assert compute.device_for(compute.GPU_FROM - 1) == "cpu"
    monkeypatch.delenv(compute.SETTING)
    with pytest.raises(RuntimeError, match=compute.SETTING):
        compute.device_for(1)
