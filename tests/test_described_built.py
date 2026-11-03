import gc
import logging
import time
from pathlib import Path

import pytest
from described_sources import WRITERS, written
from support import pipelined
from test_described import ROWS, read

from conftest import Inline
from graphspec import GENERATORS, SCRIPTED, Builder, BuildPlan, Invalid, validate
from graphspec.budgets import Refused
from graphspec.infra.files import read_json
from graphspec.operations import parameters
from tuning import build_bank, configuration, fit, open_bank
from tuning.families.heads import NONE, REALISATION
from tuning.families.model_spec import configured, groups

LOG = logging.getLogger(__name__)
pytestmark = pytest.mark.described
ROOT = Path(__file__).parent / "described"
PLAN = BuildPlan(64 << 20, 1)
BUDGET = 0.5
STATED = {"time/steps": "tasks", "time/scheme": "state", "time/several": "last", "graphs/join": "separate",
          "time/history": 1, "time/context": 0}
DEFECTS = {
    "15_drugs.py": (Invalid, "held graphs: nodes(held=) and inside= are refused by graphspec/build/mapping_build.py, "
                             "and builder.py and writing.py state no held graphs, though the contract holds them"),
    "22_dispatch.py": (Refused, "the script states split=\"'train'\", one split, so no fit can name a split to score; "
                                "the owner decides between a split to score in the script and building alone"),
    "24_tutoring.py": (Invalid, "previous(answered) is marked known from the answer of its own row, not of the row "
                                "before, so at=\"shown\" is refused as dated by what is known later"),
}
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


@pytest.mark.parametrize("name", [
    pytest.param(name, marks=pytest.mark.xfail(strict=True, raises=DEFECTS[name][0], reason=DEFECTS[name][1]))
    if name in DEFECTS else name for name in sorted(WRITERS)])
def test_a_described_script_validates_builds_and_fits_from_synthetic_sources_within_its_budget(
    name, tmp_path, monkeypatch, warmed
):
    monkeypatch.setattr("tuning.batches.packing.Packer", Inline)
    written(name, tmp_path / "sources")
    held = set(SCRIPTED.names())
    try:
        seconds = run(name, tmp_path)
    finally:
        for added in set(SCRIPTED.names()) - held:
            del SCRIPTED.held[added], GENERATORS.held[added]
    saved = read_json(tmp_path / "run" / "result.json")
    assert saved["status"] == "complete" and 0 < saved["optimizer_updates"] <= 20
    assert read(name, (ROOT / name).read_text(encoding="utf-8"))["statements"] <= ROWS[name][1]
    assert seconds <= BUDGET, (name, seconds)
