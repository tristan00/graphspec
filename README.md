**graphspec**

Build a static or temporal truth graph, derive graph variants, and tune graph transforms and GNN components together. Node types and relations come from the input schema. What is predicted is stated as tasks: a task has a kind, a time, named references to nodes, edges, sets of nodes, paths, subgraphs or the graph, and a target of a registered kind, shipped a number, a path of many `(tau, series, y)` points, a class among classes the schema states, or a set of such classes, each point predicted by the head of its kind from the task's embedding and a learned embedding of the series, for a path with an encoding of the elapsed time `tau`, and for a class as one point per class; one bank may hold kinds of task with different kinds of target. A kind of task states a utility, the value of an outcome. A fit and a search run on the device their statement names, `cpu` or `cuda`, and stop with an error when it names a GPU and none is present.

**Install and storage**

Set `GRAPHSPEC_HOME` to an explicit absolute directory before starting Python. There is no default home or profile settings file. Runtime output must stay outside the checkout and installed package tree; on Windows, every output must resolve to the same volume as `GRAPHSPEC_HOME`. Inputs may be read from other volumes. Inputs and completed banks are read-only; derived caches and temporary files live under the runtime home.

Create a virtual environment outside the checkout using the [environment setup](docs/loader.md#environment-setup). From an external build copy of the source, install with `python -B -m pip install .`, or install a built wheel. This keeps packaging metadata and build directories outside the working checkout too. Dependencies and extras are declared in `pyproject.toml`. Use `.[ladybug]` for database inspection/export, and `.[dev]` for development. Infrastructure ships in `graphspec.infra`; no sibling repository is required. The console command is `graphspec`, also available as `python -B -m tuning`.

The examples use `D:/graphspec-runtime` and its external virtual environment at `environment/Scripts/python.exe`. From the checkout, `scripts/graphspec.ps1 -Home ABSOLUTE_PATH` configures the environment before starting Python and forwards the remaining graphspec arguments. Choose your own runtime directory; the library does not assume a drive or machine path. For direct Python use, set `GRAPHSPEC_HOME` in the launching environment and use `-B` or `PYTHONDONTWRITEBYTECODE=1`. See [storage and loader details](docs/loader.md).

**One input path for different domains**

Source data → loader script → public Stage Parquet tables → bank → transformed variants → model/search.

There is no required database round trip. LadybugDB remains an optional way to query and inspect the same graph; it is not required to create Parquet files. `Stage` writes files and manifests, and `GraphTables` manages contract tables and identifiers. [The loader walkthrough](docs/loader.md) shows a complete custom graph in Python.

```powershell
./scripts/graphspec.ps1 -Home D:/graphspec-runtime synthetic --output D:/graphspec-runtime/datasets/demo --name demo --shape uniform --columns 4 --signals 2 --graphs 4 --entities 40 --steps 12 --degree 2.0 --max-degree 6 --alive 0.8 --lifetime 4.0 --period 3 --level task --seed 1
./scripts/graphspec.ps1 -Home D:/graphspec-runtime inspect D:/graphspec-runtime/datasets/demo/demo.task.json
./scripts/graphspec.ps1 -Home D:/graphspec-runtime build --task D:/graphspec-runtime/datasets/demo/demo.task.json --output D:/graphspec-runtime/banks/demo --trained train --workers 1
./scripts/graphspec.ps1 -Home D:/graphspec-runtime check D:/graphspec-runtime/banks/demo --output D:/graphspec-runtime/checks-demo.json --device cpu --seed 1 --trained train --scored validation --sees train= validation= --loss number=squared --pipeline D:/graphspec-runtime/pipelines/demo.json --settings D:/graphspec-runtime/pipelines/demo.settings.json
./scripts/graphspec.ps1 -Home D:/graphspec-runtime search run --bank D:/graphspec-runtime/banks/demo --output D:/graphspec-runtime/runs/demo --device cuda --seed 1 --seconds 600 --preparation-seconds 300 --workers 4 --variant-gb 64 --batch-sequences 64 --trained train --scored validation --sees train= validation= --loss number=squared --measures mse r2 --continuous --selector botorch --startup-trials 10 --mix random=25 tpe=25 forest=25 botorch=25 --random-params model/view --pipeline D:/graphspec-runtime/pipelines/demo.template.json
```

No command has a default for a choice of the user. A fit and a search state the splits they train on, the splits they score on and, for each of them after `--sees`, the rules of what its tasks may see, none after the `=` for a split that sees every fact; a fit and a search state, after `--loss` as `kind=loss` pairs, the registered loss of every kind of target of the bank, whose sum they train with, and the registered measures they record, each reading a kind of target of the bank, every kind read by one of them, the first of which, lower being better, ranks the trials of a search. A new search states its selector, its number of random startup trials, its mix, its random parameters and its pipeline, a JSON list of the operations it may apply in order, such as `[{"operation": "graphspec.window"}, {"operation": "graphspec.edges"}]`; the search draws which of them are present and their parameters. The example states 10 random trials and then an equal mix of random, TPE, forest and BoTorch. `--random-params` with no name after it states that none is held random. A check states a pipeline with every parameter of its operations and a file of every setting its models read, as [the contract](docs/contract.md) states under Pipelines.

Replace the Windows paths with your external workspace. `tuning` also exposes `build_bank`, `open_bank`, `configuration` and `fit`. New bank and fit output directories must not already exist. Searches resume when their existing run directory is supplied with the statement they were started with. Put a STOP file in a run to stop after its current trial.

On Windows, user-SID named mutexes enforce one resource-intensive build, check, fit, or search across runtime homes. These singleton guards create no lock files.

To inspect a graph in LadybugDB, install the extra and run:

```powershell
./scripts/graphspec.ps1 -Home D:/graphspec-runtime database --staging D:/graphspec-runtime/datasets/demo/demo.staging --output D:/graphspec-runtime/datasets/demo/demo.lbug
```

The exporter derives its schema from Parquet. Database-only tasks remain supported; when a task names staging, missing staging is an error.

**Reference**

- [Graph table contract](docs/contract.md): schema, time, causality, labels, tasks.
- [Public loader interface](docs/loader.md): graphs, shards, database export, storage.
- [Graph operations and settings](graphspec/README.md).
- [Models, packing, search, and run outputs](tuning/README.md).



The repository was extracted from tw_stack at commit `3e08d6e` in that repository. No commit hash is shared with it.
