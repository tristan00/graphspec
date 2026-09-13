**graphspec**

graphspec searches graph transforms and temporal graph models together for the best predictor of a target you
choose. You store one temporal truth graph in a LadybugDB database and name the target in a small task file.
graphspec reads node types, relations and feature widths from the database schema and parses the graph once into
a bank. Each search trial then proposes model parameters together with a graph-transform spec. The spec
derives a variant of the truth graph: a reduction, a densification or a different temporal edge scheme. The
model trains on that variant on a CUDA GPU.

- `graphspec/` — the domain-free core:
  - the database contract: schema and reader (`graphdb.py`)
  - the relation vocabulary (`vocabulary.py`, `registry.py`, `declarations.py`)
  - the parsed truth graph (`truth.py`)
  - the spec space (`spec.py`)
  - the graph operators and model representations (`operators.py`, `representations.py`, `transform.py`)

  See [graphspec/README.md](graphspec/README.md).
- `tuning/` — bank building, the model space and its components, batching and the search runner. See
  [tuning/README.md](tuning/README.md).
- `dashboard/` — a read-only results server for a running search.

Everything that is mechanism rather than graph logic lives in the sibling `graph_tuning_infra` repository: files and arrays,
LadybugDB access, process supervision, logging, Optuna plumbing, the HTTP server, hashing and the feature
encoder. `graph_tuning_infra` imports nothing from this repository; this repository imports `graph_tuning_infra`.

**Install**

Clone `graph_tuning_infra` next to this repository, then from this directory:

```powershell
pip install -e ../graph_tuning_infra
pip install -r requirements.txt
```

with a CUDA build of PyTorch; fits run on CUDA only.

On Windows the LadybugDB wheel links `libssl-3-x64.dll` and `libcrypto-3-x64.dll` without shipping them, so
OpenSSL 3 is installed system-wide (`choco install openssl.light`) or both DLLs sit in `site-packages\ladybug.libs`.

**Using your own graph**

1. Create a LadybugDB database that follows the contract below, and load your rows into it.
2. Write a task file that names the database and the target table.
3. Build a bank. The database is read once; training never touches it.
4. Check that every model runs on the bank.
5. Run the search and watch it in the dashboard.

```powershell
python -m tuning.build --task artifacts/graphdb/retention.task.json --output artifacts/bank_retention --workers 8
python -m tuning.check_models artifacts/bank_retention --output artifacts/checks_retention.json
python -m tuning.run run --bank artifacts/bank_retention --output artifacts/run_retention --continuous --selector tpe
python -m dashboard.server --directory artifacts/run_retention --port 8766
```

`retention` is the example used throughout this page.

**Synthetic data**

Development and benchmarking use generated databases, never client data. `python -m graphspec.synthetic --output artifacts/graphdb --name dev --graphs 4 --entities 2000 --steps 48` writes `dev.lbug` and `dev.task.json` following the contract: four node types, five relations with bounded degrees (`--degree`, `--max-degree`), relations that persist for a geometric lifetime and end with a removal row (`--lifetime`), observation coverage `--alive`, one query per step with categories and `--signals` channels, periods of `--period` steps, and a target on queries, nodes or graphs (`--level`). Every run is fixed by `--seed`. The generator refuses settings that cannot give two training and two validation label groups: at least ten steps for query targets, labelled periods on both sides of the 80% split for node targets, four graphs for graph targets.

**Why LadybugDB**

Graphs live in [LadybugDB](https://github.com/LadybugDB/ladybug), an embedded Cypher property-graph database
forked from Kùzu, pinned at 0.20.4. It was chosen on 10 September 2026 after comparing embedded and server
graph databases; the embedded candidates were installed and measured on this workstation.

What it gives:

- A schema that is a real declaration. Node tables are primary-keyed, relationship tables list every allowed
  FROM/TO pair, properties are typed, and fixed-width float arrays are supported. graphspec reads the schema
  back instead of asking for a separate spec.
- In-process use from a CPython 3.14 Windows wheel, with no server and one database file. Several read-only
  processes can open one database at once, which is how the bank builder parallelises.
- Fast bulk paths where training needs them: relationships imported at 1.3–3.6 million rows/s, and every read
  returns Arrow.

Reads start from the entity tables (`WHERE e.graph = $graph`), because a filter on a property of a large table
scans the whole table.

Alternatives: DuckDB with DuckPGQ was the runner-up. It was the fastest and smallest, but it stores plain
tables, and its property-graph extension is a research project with no Windows build for DuckDB 1.5.5. Kùzu is
archived and has no CPython 3.14 Windows wheel, RyuGraph is dormant, and Grafeo cannot import Parquet or
Arrow. Neo4j reserves typed graph schemas and Arrow export for Enterprise, and Apache AGE has untyped
properties. Memgraph, TigerGraph, ArangoDB, NebulaGraph, JanusGraph, Dgraph and Neptune need Docker, Linux or a
cloud account here, or do not enforce relationship endpoints.

**The database contract**

A database holds one or more graphs. A graph is one connected history: a whole book of business, a campaign, a
portfolio. One graph is enough, because labels on its nodes carry their own step and split. Time inside a graph is
a sequence of steps 0, 1, 2, … Each step holds the observations and relationship facts known at that step, and
exactly one query when the database has queries.

| Table | Kind | Columns | Meaning |
|---|---|---|---|
| `graph` | node | `id INT32` key, `name STRING` | One row per graph. Names are unique. |
| one per node type, such as `customer` | node | `id INT64` key, `graph INT32`, `entity INT32`, `key STRING` | The entities of each graph. `entity` is unique within its graph; only uniqueness matters. `key` is a readable label. |
| `observation` | node | `id INT64` key, `seq INT32`, `step INT32`, `x FLOAT[F]` | An entity's feature vector at a step. |
| `observed` | relationship | FROM every node type TO `observation` | Links each observation to its entity. |
| `query` | node, optional | `id INT64` key, `graph INT32`, `step INT32`, `period INT32`, `key INT64`, `category STRING`, `x FLOAT[F]`, optionally `signal FLOAT[S]` and `signal_observed FLOAT[S]` | Decision moments, when the history has them. |
| `query_actor`, `query_subject`, `query_context` | relationship, with `query` | FROM `query` TO node types | The three entities each query is about. |
| `edge_<relation>` | relationship | FROM/TO pairs between node types; `seq INT32`, `step INT32`, `present BOOLEAN`; any `FLOAT` attributes | Relationship facts. |
| `<target>` | node | `id INT64` key, `y DOUBLE`, `split STRING`, and `step INT32` for node targets | Labels; the name is yours. |
| `<target>_of` | relationship | FROM `<target>` TO node types, TO `query`, or TO `graph` | What each label belongs to. |

Structure:

- Node types are the source tables of `observed`, in creation order. Relations are the `edge_` tables, also in
  creation order. Order sets the relation codes and the fingerprint, so recreating tables in another order
  builds a different bank.
- The `edge_` prefix exists because node and relationship tables share one case-insensitive namespace.
  Every relationship column other than `seq`, `step` and `present` is an attribute and must be `FLOAT`.
- Relation names may not contain `:` or be `diffusion`, `rewire`, `temporal` or `untyped`; those belong to
  derived relations. Node types may not be
  named `graph`, `observation`, `observed`, `query`, `query_actor`, `query_subject`, `query_context` or `step`, and may
  not start with `edge_` or contain `:`.
- `observation.x` and `query.x` share one even width `F`. The first half holds values and the second half
  holds presence flags; `graph_tuning_infra.features.encoder(F // 2, numbers, categories)` builds such vectors by
  feature hashing. The bank stores them at float16 precision.
- `id` keys are unique across a whole table, not per graph.

Time and causality:

- With queries, query steps in each graph number 0 to n−1 with no gaps, and every graph has at least one query.
  Without queries, a graph's steps run from 0 to its last observation, fact or label, and a step may be empty.
- An observation or fact at step `s` is visible to a query or node label at step `s` and every later one, never
  to an earlier one.
- `seq` orders rows within a step. `(step, seq)` must be unique among observations, among present facts across
  all relations, and among absent facts.
- An entity has at most one observation per step. A present fact needs observations of both of its entities at
  its step, a query needs observations of its actor, subject and context at its step, and a node label needs
  an observation of its node at its step. An unchanged vector adds nothing to the event history, which records
  an observation only when `x` changes.
- A present fact holds at its own step only: the snapshot at step `s` contains exactly the facts stated at
  step `s`, so a relation that persists is restated at every step. The event history records a relation when
  it becomes present, and a removal when a `present = false` row arrives. A removal needs both entities
  observed at or before its step.
- Each query has exactly one `query_actor`, one `query_subject` and one `query_context` edge.
- `period` groups query steps into turns — a game turn, a billing cycle, a day — and never decreases. Without
  queries every step is its own period. A spec keeps these periods or replaces them with fixed-length ones.
- `key` is the query's own identifier, kept as `decision` in the bank. `category` names the kind of query;
  the sorted distinct categories form the vocabulary for affinity features.
- `signal` and `signal_observed` are optional per-query channels with an observed mask. The signal model is
  offered only when they exist and the target is on queries or graphs.

**Example schema**

A renewal model for insurance customers. Each graph is one customer's history; each step is a month.

```cypher
CREATE NODE TABLE graph(id INT32 PRIMARY KEY, name STRING);
CREATE NODE TABLE customer(id INT64 PRIMARY KEY, graph INT32, entity INT32, key STRING);
CREATE NODE TABLE policy(id INT64 PRIMARY KEY, graph INT32, entity INT32, key STRING);
CREATE NODE TABLE agent(id INT64 PRIMARY KEY, graph INT32, entity INT32, key STRING);
CREATE NODE TABLE region(id INT64 PRIMARY KEY, graph INT32, entity INT32, key STRING);
CREATE NODE TABLE observation(id INT64 PRIMARY KEY, seq INT32, step INT32, x FLOAT[128]);
CREATE NODE TABLE query(id INT64 PRIMARY KEY, graph INT32, step INT32, period INT32, key INT64, category STRING, x FLOAT[128]);
CREATE NODE TABLE renewal(id INT64 PRIMARY KEY, y DOUBLE, split STRING);
CREATE REL TABLE observed(FROM customer TO observation, FROM policy TO observation, FROM agent TO observation, FROM region TO observation);
CREATE REL TABLE query_actor(FROM query TO policy);
CREATE REL TABLE query_subject(FROM query TO customer);
CREATE REL TABLE query_context(FROM query TO agent);
CREATE REL TABLE edge_holds(FROM customer TO policy, seq INT32, step INT32, present BOOLEAN);
CREATE REL TABLE edge_sold_by(FROM policy TO agent, seq INT32, step INT32, present BOOLEAN, commission FLOAT);
CREATE REL TABLE edge_located_in(FROM customer TO region, FROM agent TO region, seq INT32, step INT32, present BOOLEAN);
CREATE REL TABLE renewal_of(FROM renewal TO query);
```

Load node tables first, then relationships, with `COPY` from Parquet. A relationship file starts with the source
and target `id` columns, followed by the properties in declared order. Load one file per FROM/TO pair and name
the pair, as `graph_tuning_infra.database.copy` does:

```cypher
COPY customer FROM 'customer.parquet';
COPY observation FROM 'observation.parquet';
COPY observed FROM 'observed_customer.parquet' (from='customer', to='observation');
COPY edge_located_in FROM 'located_in_agent.parquet' (from='agent', to='region');
```

**Targets on nodes, queries or graphs**

A target is any real value: a reward, an outcome, an amount. Keep the labels in a node table of your choosing,
attach each one with `<target>_of`, and name the table in the task file. The destination of `<target>_of`
decides the level.

- **On nodes** (`FROM renewal TO customer`, to one or more node types): the target table also has `step INT32`.
  Each label is predicted for its node at that step from everything visible then, so the node needs an
  observation at that step. A node can carry labels at several steps, one per step. Queries are optional.
- **On a query** (`FROM renewal TO query`): one label per query, predicted at that query's step from
  everything visible then. Queries without a label still feed the history.
- **On a graph** (`FROM renewal TO graph`): one label per graph, predicted at the graph's last query.

Query and graph targets need the query tables. A single graph with node targets needs no query tables, only
the target table and its relationship next to the tables above:

```cypher
CREATE NODE TABLE renewal(id INT64 PRIMARY KEY, y DOUBLE, split STRING, step INT32);
CREATE REL TABLE renewal_of(FROM renewal TO customer);
```

For every level:

- `split` is `train` or `validation`, set per label.
- A node carries at most one label per step, a query or graph at most one label, every graph at least one, and
  every `y` is finite at float32 precision.
- Building needs at least two training labels and two validation labels that differ at float32 precision.
- Training targets are standardised with the training mean and sample standard deviation. Fits are scored by
  raw MSE and R² on the whole validation split.

No spec removes or filters the node types a node target labels, or an observation a query needs at its step.

**Task file**

```json
{
  "name": "retention",
  "database": "retention.lbug",
  "target": "renewal"
}
```

| Key | Meaning |
|---|---|
| `name` | The registry name, part of the fingerprint. |
| `database` | Database path, relative to the task file. |
| `target` | The target table; `<target>_of` must exist. |

All three keys are required and no others are accepted. Everything else a variant needs comes from the spec
a trial proposes.

**Building a bank**

`python -m tuning.build --task TASK --output BANK --workers N`. The output directory must not exist.

1. The builder reads the schema, derives the relation vocabulary and refuses reserved names.
2. It reads the labels, applies the target rules above and writes `supervision.json`.
3. Worker processes open the database read-only, each with a 4 GiB buffer pool. Each reads one graph with
   anchored Cypher, checks it against the contract and writes its truth arrays.
4. It writes `manifest.json`. That pins the declaration and fingerprint, the task, the target level,
   categories, signal width and per-graph sizes.

Variants are not built here. A trial's spec materialises its variant of a graph on first use, under
`variants/`.

```
BANK/
  manifest.json                     declaration, fingerprint, task, target level, categories, graphs
  supervision.json                  labels, split indices, training mean and standard deviation
  truth/graph_0000/                 the truth graph: one .npy per array and graph.json
  variants/<digest>/variant.json    a spec, its representation groups and the graphspec and graph_tuning_infra source hash
  variants/<digest>/graph_0000/     that variant's arrays for graph 0
```

**Running the search**

```powershell
python -m tuning.run run --bank BANK --output RUN --continuous --selector tpe
python -m tuning.run report RUN
```

- `--continuous` keeps proposing trials until a file named `STOP` appears in `RUN`; the current fit finishes
  first.
- `--selector tpe`, the default, proposes the model, its parameters and its graph spec with Optuna TPE.
  `--selector random` samples all three uniformly.
- `--seconds` sets each fit's budget (600), `--batch-sequences` the sequences per batch at history 16 (64), and
  `--seed` the proposal seed.
- `--workers` (4) builds the trial's variant for every graph in parallel before the worker starts. `--variant-gb` (300)
  is the disk budget for `variants/`; the least recently used variants are removed first.
- `--resume` continues a run with its planned selector, seed, budget and batch size, and refuses a rebuilt or
  different bank or declaration.

`python -m dashboard.server --directory RUN --port 8766` serves every attempt at http://127.0.0.1:8766/,
sorted by validation MSE and refreshed every ten seconds. It only reads run files.

**Provenance**

History was path-filtered out of tw_stack, so no commit hash is shared with it. The extraction point was
tw_stack commit 3e08d6e, which exists in that repo, not this one.
