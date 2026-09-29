**Building a graph without a database**

`graphspec.Stage` is the public Parquet writer. It writes the table manifest and shards; users do not need to reproduce file naming or internal identifiers. `graphspec.GraphTables` constructs contract tables from named nodes, observations, edges, and node labels. Temporal query tables are written through the same `Stage` interface.

Set an absolute `GRAPHSPEC_HOME` before launching the Python process. For example, an external Windows environment can run a saved loader with:

```powershell
$env:GRAPHSPEC_HOME = 'D:/graphspec-runtime'
& "$env:GRAPHSPEC_HOME/environment/Scripts/python.exe" -B ./my_loader.py
```

```python
import os
from pathlib import Path
from graphspec import GraphTables, Stage, write_task, build_bank
import numpy as np

runtime = Path(os.environ["GRAPHSPEC_HOME"])
workspace = runtime / "datasets" / "example"
staging = workspace / "example.staging"
graph = GraphTables(index=0, name="customers", features=4)
graph.nodes("client", ["a", "b", "c", "d"])
graph.nodes("branch", ["north"])
graph.observe("client", ["a", "b", "c", "d"], np.ones((4, 4)))
graph.observe("branch", ["north"], np.ones((1, 4)))
graph.edges("assigned", "client", "branch", ["a", "b", "c", "d"], ["north"] * 4)
graph.labels("client", ["a", "b", "c", "d"], [0, 1, 0, 1], ["train", "train", "validation", "validation"])
with Stage(staging) as stage:
    graph.write(stage)
task = write_task(workspace / "example.task.json", "example", staging, "outcome")
build_bank(task, runtime / "banks" / "example", workers=1)
```

The first half of each feature vector contains values; the second half contains presence indicators. The example supplies values of one with all values present. `graphspec.infra.features.encoder` provides optional feature hashing. Integer identifiers in raw tables are managed by `GraphTables`; readable keys are strings. Nodes must be declared before referencing them. Each `observe`, `edges`, or `labels` call accepts a nonnegative `step`, with zero representing a static snapshot. Temporal observations and facts follow the [contract's time and causality rules](contract.md). Edge attributes are keyword arrays of finite float32 values.

Use the public configuration builder for one model fit on either adapter's bank:

```python
from graphspec import configuration, fit

bank = runtime / "banks" / "example"
config = configuration(
    bank, model={"view": "snapshot", "hidden": 16}, graph={"time/history": 1}, training={"epochs": 1}
)
result = fit(bank, config, runtime / "runs" / "example-fit", workers=1)
```

Put multiprocessing workflows inside a `main` function called under `if __name__ == "__main__":` when saving them as Python scripts. `fit` requires CUDA and scores every validation label after full training epochs. The same bank can also be passed to the search CLI; [tuning/README.md](../tuning/README.md) describes the model and graph parameter names.

For multiple graphs, use consecutive graph indices from zero. Each graph writes its own tables into the same Stage, in graph order. A source that produces its tables by query can call `Stage.write` directly with Arrow tables keyed by `(table, source_type, target_type)`. `Stage.merge` combines manifests from shards below the staging directory without copying their Parquet data.

Sources declare `staging` or `database` in a task file. A staging-only task needs `name`, `staging`, and `target`; it does not need a fictional database filename. For older task files naming both, staging is explicitly preferred and must exist. Missing staging never falls back to another source.

LadybugDB is an optional query and inspection format. Install `graphspec[ladybug]`, then export any staging with `graphspec database --staging STAGING --output DATABASE.lbug`. The exporter derives the database schema from the staged tables. There is no CSV → LadybugDB → Parquet round trip: adapters write Parquet once, bank building reads it, and database export is optional. Existing database-only task files still work.

**Runtime storage**

`GRAPHSPEC_HOME` is mandatory and must be an absolute path outside the source checkout and installed package tree. There is no profile settings lookup or default location. Set it before importing the library or launching a command. On Windows, dataset, bank, run, report, and other output paths must resolve to the configured runtime home's volume. Directory links do not bypass the path checks. Input files and completed banks may be read from other volumes.

The examples use `D:/graphspec-runtime`; any valid external runtime home can be selected. Keep staged datasets in `datasets/`, banks in `banks/`, runs in `runs/`, and the external virtual environment in `environment/`. Managed caches, including compilation and derived graph data, live under `cache/`; temporary files live under `tmp/`. The checkout launcher accepts mandatory `-Home` and sets the environment before Python starts: `./scripts/graphspec.ps1 -Home D:/graphspec-runtime search --help`.

Inputs are read-only: partition indexes and derived bank data are stored under the runtime home. A bank can be moved without changing its files; caches can be rebuilt. Parquet partition cache identity includes the manifest, file sizes, and modification times; replace edited inputs normally rather than preserving their old timestamps.

On Windows, named mutexes scoped to the user's SID serialize public bank builds, dataset adapters, model checks, fits, and searches across runtime homes. These guards create no lock files in the profile, runtime home, or checkout; Windows releases ownership when its owner exits. Internal worker processes belong to the admitted job.

For source development use `python -B` or `PYTHONDONTWRITEBYTECODE=1` to keep bytecode out of the checkout. Pytest's repository cache is disabled. Compilation caches, lint caches, and test temporary directories must stay under the configured runtime home; use an external `--basetemp` beneath its `tmp/` directory. Previous `~/.graphspec/settings.json` files are retired and are not read.

## Environment setup

Environment creation and package installation happen before the library can configure storage. Set their temporary and cache directories explicitly too:

```powershell
$env:GRAPHSPEC_HOME = 'D:/graphspec-runtime'
$env:TEMP = "$env:GRAPHSPEC_HOME/tmp"
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:PIP_CACHE_DIR = "$env:GRAPHSPEC_HOME/cache/pip"
$env:PYTHONDONTWRITEBYTECODE = '1'
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
python -B -m venv "$env:GRAPHSPEC_HOME/environment"
```

Use `environment/Scripts/python.exe` inside that home for installation. Install a wheel, or run `-B -m pip install .` from a build copy outside the checkout on the runtime volume. Subsequent graphspec commands can use `./scripts/graphspec.ps1 -Home D:/graphspec-runtime` followed by their CLI arguments.
