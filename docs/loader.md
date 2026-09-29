**Building a graph without a database**

`graphspec.Stage` is the public Parquet writer. It writes the table manifest and shards; users do not need to reproduce file naming or internal identifiers. `graphspec.GraphTables` constructs contract tables from a schema and named nodes, observations, edges and tasks whose references are one node each or the graph. Other tables of tasks are written through the same `Stage` interface.

Set an absolute `GRAPHSPEC_HOME` before launching the Python process. For example, an external Windows environment can run a saved loader with:

```powershell
$env:GRAPHSPEC_HOME = 'D:/graphspec-runtime'
& "$env:GRAPHSPEC_HOME/environment/Scripts/python.exe" -B ./my_loader.py
```

```python
import os
from pathlib import Path
from graphspec import GraphTables, Schema, Stage, write_task
from tuning import build_bank

runtime = Path(os.environ["GRAPHSPEC_HOME"])
workspace = runtime / "datasets" / "example"
staging = workspace / "example.staging"
schema = Schema(
    dict(
        version=3,
        time=dict(model="clock", units=["day"], ties="key"),
        observers=False,
        columns=dict(
            balance=dict(type="number", encoder="standard"),
            tier=dict(type="category", encoder="onehot"),
            open=dict(type="flag", encoder="value"),
        ),
        graph=dict(columns=[]),
        nodes=dict(client=dict(columns=["balance", "tier"]), branch=dict(columns=["open"])),
        edges=dict(
            assigned=dict(endpoints=[["client", "branch"]], directed=True, columns=[], relation="declared")
        ),
        tasks=dict(
            key="number",
            grouping=None,
            signals=0,
            references=dict(holder=dict(sort="node", kinds=["client"])),
            kinds=dict(
                outcome=dict(
                    references=["holder"],
                    columns=[],
                    target=dict(kind="number", series="outcome"),
                    weight=False,
                    absent="stop",
                    event=None,
                    matures=None,
                )
            ),
        ),
    )
)
graph = GraphTables(index=0, name="customers", unit="day", schema=schema)
always = dict(since=0, until=None)
graph.nodes("client", ["a", "b", "c", "d"])
graph.nodes("branch", ["north"])
graph.observe("client", ["a", "b", "c", "d"], always, balance=[1.0, 2.5, None, 4.0], tier=["low", "high", "low", None])
graph.observe("branch", ["north"], always, open=[True])
graph.edges("assigned", "client", "branch", ["a", "b", "c", "d"], ["north"] * 4, always, present=True)
graph.tasks(
    "outcome",
    [1, 2, 3, 4],
    0,
    dict(holder=("client", ["a", "b", "c", "d"])),
    ["train", "train", "validation", "validation"],
    [0, 1, 0, 1],
    [None, None, None, None],
    None,
    None,
)
with Stage(staging) as stage:
    graph.write(stage)
task = write_task(workspace / "example.task.json", "example", staging, schema)
build_bank(task, runtime / "banks" / "example", workers=1, trained=("train",))
```

The schema states every kind of node, every edge table and every column with its type and encoder; [the contract](contract.md) describes it. Each kind has its own columns, and `None` states a missing value. Integer identifiers in raw tables are managed by `GraphTables`; readable keys are strings. Nodes must be declared before referencing them. A graph states its unit of time, one of those of the schema, or `None` where the schema has no clock. Each `observe` and `edges` call states when its facts hold as a mapping of exactly the columns of time the schema states, each one value for every row or a list with one value per row; a missing `until` states no end. `tasks` states, in this order, the kind of its tasks, their keys, their time, under each reference of the kind the kind of node and the keys it names, their splits, their targets, the times from which the targets are known, their weights and their observers, each of the last four as a list or as `None` where the kind or the schema states none. Rows stated in a shorthand form are reduced to such facts by `graphspec.forms.reduced`. Facts and views follow the [contract's rules of time](contract.md). The values of edges are keyword lists named as the columns of their edge table; an edge table that takes its relations from a column is given them as `relation`.

Use the public configuration builder for one model fit on a bank:

```python
from tuning import configuration, fit

bank = runtime / "banks" / "example"
statement = dict(
    version=1,
    device="cpu",
    seed=1,
    seconds=600,
    preparation_seconds=300,
    workers=1,
    variant_cache_bytes=1 << 30,
    batch_sequences=64,
    trained=["train"],
    scored=["validation"],
    sees=dict(train=[], validation=[]),
)
config = configuration(
    bank,
    model={"view": "snapshot", "hidden": 16},
    graph={"time/history": 1},
    training={"epochs": 1},
    seed=statement["seed"],
    splits={name: statement[name] for name in ("trained", "scored", "sees")},
)
result = fit(bank, config, runtime / "runs" / "example-fit", statement)
```

Put multiprocessing workflows inside a `main` function called under `if __name__ == "__main__":` when saving them as Python scripts. `fit` takes its statement, in which every field is required and an absent field stops the fit with an error naming it; it runs on the device the statement names, trains on the splits the statement names and scores every target of the splits it names for scoring after full training epochs. The same bank can also be passed to the search CLI; [tuning/README.md](../tuning/README.md) describes the model and graph parameter names.

For multiple graphs, use consecutive graph indices from zero. Each graph writes its own tables into the same Stage, in graph order. A source that produces its tables by query can call `Stage.write` directly with Arrow tables keyed by `(table, source_type, target_type)`. `Stage.merge` combines manifests from shards below the staging directory without copying their Parquet data.

Sources declare `staging` or `database` in a task file. A staging-only task needs `name`, `staging` and `schema`; it does not need a fictional database filename. For older task files naming both, staging is explicitly preferred and must exist. Missing staging never falls back to another source.

LadybugDB is an optional query and inspection format. Install `graphspec[ladybug]`, then export any staging with `graphspec database --staging STAGING --output DATABASE.lbug`. The exporter derives the tables of the database from the staged tables; a task that reads the database states the same schema. There is no CSV → LadybugDB → Parquet round trip: a loader writes Parquet once, bank building reads it, and database export is optional. Existing database-only task files still work.

**Runtime storage**

`GRAPHSPEC_HOME` is mandatory and must be an absolute path outside the source checkout and installed package tree. There is no profile settings lookup or default location. Set it before importing the library or launching a command. On Windows, dataset, bank, run, report, and other output paths must resolve to the configured runtime home's volume. Directory links do not bypass the path checks. Input files and completed banks may be read from other volumes.

The examples use `D:/graphspec-runtime`; any valid external runtime home can be selected. Keep staged datasets in `datasets/`, banks in `banks/`, runs in `runs/`, and the external virtual environment in `environment/`. Managed caches, including compilation and derived graph data, live under `cache/`; temporary files live under `tmp/`. The checkout launcher accepts mandatory `-Home` and sets the environment before Python starts: `./scripts/graphspec.ps1 -Home D:/graphspec-runtime search --help`.

Inputs are read-only: partition indexes and derived bank data are stored under the runtime home. A bank can be moved without changing its files; caches can be rebuilt. Parquet partition cache identity includes the manifest, file sizes, and modification times; replace edited inputs normally rather than preserving their old timestamps.

On Windows, named mutexes scoped to the user's SID serialize public bank builds, model checks, fits, and searches across runtime homes. These guards create no lock files in the profile, runtime home, or checkout; Windows releases ownership when its owner exits. Internal worker processes belong to the admitted job.

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
