import gc
import logging
import re
import runpy
import time
from pathlib import Path

import pytest
from described_sources import WRITERS, csv, day, written
from support import pipelined
from test_described import REASONS, ROWS, read

from conftest import Inline
from graphspec import GENERATORS, SCRIPTED, Builder, BuildPlan, inspect, validate
from graphspec.infra.files import read_json
from graphspec.operations import parameters
from tuning import build_bank, configuration, fit, open_bank
from tuning.families.heads import NONE, REALISATION
from tuning.families.model_spec import configured, groups

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.described
ROOT = Path(__file__).parent / "described"
WALKTHROUGH = Path(__file__).parent / "walkthrough"
DOCUMENT = Path(__file__).parents[1] / "docs" / "loader.md"
PLAN = BuildPlan(64 << 20, 1)
BUDGET = 0.5
STATED = {"time/steps": "tasks", "time/scheme": "state", "time/several": "last", "graphs/join": "separate",
          "time/history": 1, "time/context": 0}
MEASURED = {"number": ("squared", "mse"), "path": ("squared", "mse"), "decision": ("squared", "mse"),
            "class": ("entropy", "accuracy"), "link": ("entropy", "reciprocal"), "set": ("entropy", "reciprocal"),
            "classes": ("binary", "binary"), "sequence": ("tokens", "tokens"), "survival": ("hazard", "brier")}


def drawn(schema):
    started = time.perf_counter()
    found = []
    for entry in schema.tasks["kinds"].values():
        target = entry["target"] or {}
        if target.get("kind") in ("link", "set"):
            parts = ["link", target["relation"]] if target["kind"] == "link" else ["set", target["sets"]]
            found.append(":".join(parts + ([target["about"]] if target["about"] else [])))
    LOG.debug("drawn exit found=%d seconds=%.4f", len(found), time.perf_counter() - started)
    return found


def stated(directory, splits):
    started = time.perf_counter()
    registry = open_bank(directory).registry
    schema = registry.schema
    model = configured(registry, dict(view="snapshot", hidden=16, **({REALISATION: NONE} if schema.decided() else {})))
    names, values = ["graphspec.window"], dict(STATED, **{"sets/%s/storage" % kind: "node" for kind in schema.sets})
    for of in drawn(schema):
        names.append("graphspec.negatives")
        values.update({"negatives/of": of, "negatives/rule": "uniform", "negatives/count": 2})
    pipeline, rest = pipelined(registry, values, names)
    graph = parameters.draw(parameters.settings(registry, groups(model)),
                            lambda parameter: parameter.check(rest[parameter.name]) if parameter.name in rest
                            else parameter.given({}))
    config = configuration(directory, model=model, training={"epochs": 1}, pipeline=pipeline, graph=graph, seed=1,
                           splits=splits)
    kinds = list(schema.targets())
    statement = dict(
        version=5, device="cpu", seed=1, derivation_seed=1, seconds=60, preparation_seconds=60, workers=1,
        variant_cache_bytes=1 << 30, batch_sequences=16, loss={kind: MEASURED[kind][0] for kind in kinds},
        measures=sorted({MEASURED[kind][1] for kind in kinds}), alike=1.0, **splits,
    )
    LOG.debug("stated exit kinds=%d seconds=%.4f", len(kinds), time.perf_counter() - started)
    return config, statement


def run(name, folder):
    gc.collect()
    started, spent = time.perf_counter(), time.process_time()
    trained, *scored = WRITERS[name][1]
    builder = Builder(ROOT / name, folder / "sources")
    validate(builder, PLAN)
    validated = time.perf_counter()
    built = build_bank(builder, folder / "bank", 1, [trained], plan=PLAN)
    banked = time.perf_counter()
    LOG.info("described built script=%s splits=%s", name, built["split_rows"])
    splits = dict(trained=[trained], scored=scored, sees={split: [] for split in [trained, *scored]})
    config, statement = stated(folder / "bank", splits)
    fit(folder / "bank", config, folder / "run", statement)
    seconds = time.process_time() - spent
    LOG.info("described run exit script=%s validate=%.3f build=%.3f fit=%.3f seconds=%.3f processor=%.3f", name,
             validated - started, banked - validated, time.perf_counter() - banked, time.perf_counter() - started,
             seconds)
    return seconds


@pytest.fixture(scope="module")
def warmed(tmp_path_factory):
    folder = tmp_path_factory.mktemp("warmed")
    written("02_interactions.py", folder / "sources")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("tuning.batches.packing.Packer", Inline)
        return run("02_interactions.py", folder)


@pytest.mark.parametrize("name", sorted(WRITERS))
def test_a_described_script_validates_builds_and_fits_from_synthetic_sources_within_its_budget(
    name, tmp_path, monkeypatch, warmed
):
    monkeypatch.setattr("tuning.batches.packing.Packer", Inline)
    written(name, tmp_path / "sources")
    held = set(SCRIPTED.names())
    try:
        seconds = run(name, tmp_path)
        builder = Builder(ROOT / name, tmp_path / "sources")
        assert builder.functions == REASONS.get(name, {})
        if builder.functions:
            shown = inspect(builder, plan=PLAN)
            assert all("function %s: %s" % pair in shown for pair in builder.functions.items())
    finally:
        for added in set(SCRIPTED.names()) - held:
            del SCRIPTED.held[added], GENERATORS.held[added]
    saved = read_json(tmp_path / "run" / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20
    assert read(name, (ROOT / name).read_text(encoding="utf-8"))["statements"] <= ROWS[name][1]
    assert seconds <= BUDGET, (name, seconds)


def library(folder):
    started = time.perf_counter()
    folder.mkdir(parents=True)
    csv(folder, "members.csv", [dict(member="m%d" % at, joined=day(at), branch="b%d" % (at % 2)) for at in range(6)])
    loans = []
    for at in range(40):
        kept = 3 + (at * 7) % 13
        loans.append(dict(loan="l%d" % at, member="m%d" % (at % 6), book="k%d" % (at % 7), lent=day(10 + at),
                          back=day(10 + at + kept), late=str(kept > 9).lower()))
    csv(folder, "loans.csv", loans)
    LOG.debug("library exit loans=%d seconds=%.4f", len(loans), time.perf_counter() - started)


def test_the_walkthrough_of_the_document_inspects_builds_and_fits_its_script_within_the_budget(
    tmp_path, monkeypatch, warmed, capsys
):
    monkeypatch.setattr("tuning.batches.packing.Packer", Inline)
    script = (WALKTHROUGH / "library.py").read_text(encoding="utf-8")
    opening, rest = script.split("\n\n", 1)
    found = read("library.py", opening + '\n\nDESCRIPTION = "a library lends books"\n' + rest)
    assert found["refused"] == [] and found["functions"] == {} and found["statements"] == 6
    assert found["counts"] == dict(sources=2, declarations=4, functions=0)
    (tmp_path / "library.py").write_text(script, encoding="utf-8")
    library(tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GRAPHSPEC_HOME", str(tmp_path / "home"))
    gc.collect()
    spent = time.process_time()
    runpy.run_path(str(WALKTHROUGH / "run.py"), run_name="walkthrough")
    seconds = time.process_time() - spent
    LOG.info("walkthrough exit processor=%.3f", seconds)
    printed = capsys.readouterr().out
    assert "member" in printed and "overdue" in printed and "borrowed" in printed
    saved = read_json(tmp_path / "home" / "runs" / "library" / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20
    assert seconds <= BUDGET, seconds


def test_the_walkthrough_held_here_is_the_one_the_document_shows():
    if not DOCUMENT.exists():
        pytest.skip("docs/loader.md is not in this checkout; the walkthrough held under tests/walkthrough is the one run")
    blocks = re.findall(r"```python\n(.*?)```", DOCUMENT.read_text(encoding="utf-8"), re.S)
    assert blocks[:2] == [(WALKTHROUGH / name).read_text(encoding="utf-8") for name in ("library.py", "run.py")]
