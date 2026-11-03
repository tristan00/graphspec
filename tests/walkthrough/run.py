import os
from pathlib import Path

from graphspec import Builder, BuildPlan, inspect
from tuning import build_bank, configuration, fit

home = Path(os.environ["GRAPHSPEC_HOME"])
library = Builder("library.py", sources="data")
plan = BuildPlan(memory_bytes=64 << 20, workers=1)
print(inspect(library, plan=plan))
bank = home / "banks" / "library"
build_bank(library, bank, workers=1, trained=["train"], plan=plan)
splits = dict(trained=["train"], scored=["validation"], sees=dict(train=[], validation=[]))
statement = dict(
    version=5, device="cpu", seed=1, derivation_seed=1, seconds=60, preparation_seconds=60, workers=1,
    variant_cache_bytes=1 << 30, batch_sequences=16, loss={"class": "entropy"}, measures=["accuracy"],
    alike=1.0, **splits,
)
config = configuration(
    bank,
    model={"view": "snapshot", "hidden": 16},
    training={"epochs": 1},
    pipeline=[dict(operation="graphspec.window", parameters={"time/scheme": "state"})],
    graph={"time/steps": "tasks", "graphs/join": "separate", "nodes/untyped": False,
           "relations/direction": "inverse", "time/history": 1, "time/context": 0},
    seed=1,
    splits=splits,
)
fit(bank, config, home / "runs" / "library", statement)
