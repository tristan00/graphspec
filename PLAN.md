**Plan: a general graph operations and GNN tuning library**

Written 2026-09-28. Replaces the plan dated 2026-09-18.

The library builds graphs from what a user states about their data, applies graph operations to them, and tunes graph neural networks over them. It never knows a dataset. The standard for the tooling: a graph a person can describe in a few sentences takes a few lines to build, because every operation has a handler. A long script shows a missing handler, and that is a defect in the library.

Rules while working this plan:

- Anything blocked stays at the top of every status report until it clears.
- A section may not begin until every subsection before it is done. The one exception is work done autonomously while blocked on something only the owner can give.
- Every function added logs its time on exit when it is written. No comments, no docstrings.
- No fallbacks: a value that is missing, a reference that does not resolve, or a setting that was not stated stops the work with an error that names it, unless the user stated a rule for it.
- The test suite stays under 20 seconds with the 4 workers it runs with today (`pyproject.toml:28`). The cases of one subsection share one bank and one fit wherever they can. Each check inside it has a budget of 0.2 seconds unless its subsection states another; a check that fits has 0.5 seconds per fit. A check that needs a GPU or a database server is skipped inside the suite where none is present and is run in section 11. Fits inside tests run on CPU on synthetic graphs of at most 200 nodes for at most 20 steps.
- Builds, benchmarks and fits on real data are started detached, through `scripts/detached.ps1` until 10.4 replaces it, and never as a child of the agent.
- Nothing runs beside a search except downloading, reading files, and reading stored arrays with one thread. No `.py` file under `graphspec/` or `tuning/` is edited while a search runs.
- Whether an action validated or succeeded is never a model feature and never selects training data.
- A question about a graph or a model is answered from the descriptions of sections 4 and 9. A question that can only be answered by reading code is recorded as a defect of those descriptions and removed by adding to them (9.7).

**Blocked**

- B1. Every section edits `graphspec/` and `tuning/`, runs tests or uses the machine. None starts while a search is running. No search is running on 28 September at 21:00; the owner had the last one killed. A search executes the checkout (`scripts/detached.ps1` starts it with the checkout as working directory), so an edit to a `.py` file under `graphspec/` or `tuning/` reaches its next trial, and an edit to a `.py` file under `graphspec/` changes the key of every cached variant. Only the owner decides when a search stops. Editing documents, `PLAN.md`, `pyproject.toml`, `tests/` and `scripts/` does not affect a search.
- B2. Cleared on 28 September at 22:53. Every dataset named in the column Use of the table of public data is on disk, recorded with a checksum and no error, at its recorded size; the audit is `D:/graphspec-runtime/sources/audit-20260928.md`. Of the replays of a strategy game 60 of 71 packs are complete and the rest are arriving; the one pack the plan uses is complete. Sizes that were unknown: `uplift` 311 MB, `hypergraphs` 240 MB. The column On disk of that table is as of 18:47 and is replaced by the audit.

**Words used in this plan**

| Word | Meaning |
|---|---|
| fact | One statement about a graph: a node exists, an edge holds, a value is set |
| schema | The kinds of node and relation of a graph, their columns and types |
| bank | The prepared form of a set of graphs with their tasks, which fits read |
| truth graph | A graph as stored in a bank before any operation is applied |
| operation | A function that takes graphs and parameters and returns a graph |
| pipeline | An ordered list of operations with their parameters |
| variant | A graph after a pipeline has been applied |
| representation | The form in which a variant is handed to a family of models |
| task | One thing to predict: a kind, a time, references, a target |
| split | A named group of tasks; a fit states which splits it trains on and which it scores on |
| utility | A function, stated by a kind of task, from a predicted target to one number, with the direction that is better |
| decision | A task at which one option of several is chosen |
| option | What could be chosen at a decision: one action or several |
| action | A kind, an actor, targets with their roles, and parameters with their types |
| recorded, regenerated | Where an option comes from: logged as offered, or produced afterwards by a generator because only the chosen one was logged |
| realisation | The form in which an option is put before a model |
| edit | A list of additions, removals and changes of value applied to a graph over a range of time |
| segment | A stretch of a graph derived in one piece, with state carried from the stretch before |
| fork | The state of a derivation at a time, from which derivation continues separately |
| trunk | The derivation from which forks are taken |
| overlay | What a fork holds of its own, on top of what it shares with the trunk |
| physical join | Two graphs written end to end as one graph, used by tests as the reference for segments |
| series | One of several quantities predicted over time |
| family | A kind of model |
| fit | One training of one configuration |
| search | A sequence of fits, called trials, each proposed by a selector |
| selector | The method that proposes the next trial |
| candidate set | The configurations a selector scores before proposing one |
| description | What the library writes, from the data and not from its code, about what a graph holds, what a pipeline did to it, and what a model reads |
| reach | The content that is connected to a task's references at the task's time, and by which relations |
| influence | How far a model's prediction moves when one input alone is changed |
| card | The document written for every fit that answers a fixed list of questions about it |
| statement of a fit, of a search | The record that holds every choice made for it (1.6, 8.12) |
| staged | The tables a builder writes, before a bank is built from them |
| shard | One of several parts of staged output written in parallel |
| manifest | The record written beside staged output or a bank of what it holds and how it was made |
| fingerprint | A hash of content, equal when the content is equal |
| extra | An optional group of dependencies in `pyproject.toml` |
| runtime home | `D:/graphspec-runtime`, where banks, searches, caches and downloads live |

**History**

- 14 September: a first plan, `LIBRARY.md`. The plan this one replaces is dated 18 September in its text and was committed on 28 September. It named two datasets. Its builder and dataset sections were not built.
- That plan was paused because the library kept being fitted to the two datasets. A third dataset was brought in to break the assumptions the first two shared.
- The third dataset was added inside the library as an adapter written by hand (first file dated 24 September), with path targets and, on 28 September, a score named spread and measures of sequences. The work of 28 September is uncommitted and is not to be committed as it stands.
- 28 September: five reviews of the code, seven surveys of outside practice, and seven reviews of drafts of this plan. Public datasets were downloaded to develop against.
- 28 September: the owner decided that working with decisions and options is a key capability of the library, the way of scoring it not yet settled.
- 28 September: a search of 101 trials on the third dataset was stopped by the owner. Asked what its best design reads of an option, this plan's agent answered from a reading of the code, and the answer was wrong: stored predictions showed that about half of the design's differentiation came from a route the reading had called a side effect. A hand-over of that design to another project was cancelled for that reason. The library could not say what its own models read; sections 4 and 9 exist to remove that.

**State of the code, measured 2026-09-28 on the working tree**

Dataset code inside the library:

| Item | Where |
|---|---|
| Three adapter packages of 499, 2,413 and 3,140 lines | `churn/`, `vic3/`, `totalwar/` |
| Packaged, with an extra for one of them | `pyproject.toml:12`, `:24` |
| One verb per dataset | `graphspec/__main__.py:6-8` |
| Database helper used only by adapters | `graphspec/infra/postgres.py` |
| Dataset documents | `CHURN.md`, `TOTALWAR.md`, `VIC3.md` |
| Dataset tests | `tests/test_churn.py`, `tests/test_adapters.py`, `tests/test_vic3.py`, `tests/test_library.py:59` |
| Dataset names in general documents | `README.md`, `docs/loader.md`, `tuning/README.md:5` |
| Adapters import internals | `graphspec.staged.IDS`, `graphspec.staging.fixed`, `graphspec.infra.*` |
| A rule for naming the graphs of one dataset, written into the contract | `docs/contract.md:124` |
| The largest adapter's mapping is 50 hand-written loops over 73 source tables, one per table | `vic3/source.py:10-67`, `vic3/graph.py` |

Assumptions fixed in the core. All three datasets satisfy them, so none of the three can expose them.

| Assumption | Where | Removed by |
|---|---|---|
| Time is an integer step; a fact holds at its own step only and is restated at every step; an event at one instant has no form of its own | `docs/contract.md:48-51` | 2.2 |
| Periods never decrease; steps, periods and joined graphs form a fixed hierarchy of three levels | `graphspec/truth.py:450`, `graphspec/operators.py:151` | 2.2 |
| Every fact is visible to every task at or after its time | whole core | 2.2 |
| At most one observation per entity per step | `docs/contract.md:44` | 2.2 |
| Exactly one query per step when queries exist | `graphspec/truth.py:454`, `:610` | 2.3 |
| A query has exactly three roles named actor, subject, context, each observed at the query's step; nodes in a role are never filtered | `graphspec/schema.py:2`, `graphspec/truth.py:601`, `graphspec/operators.py:422-438` | 2.3 |
| Targets attach to a node, a query, a graph or a path; one level per bank; targets are numbers | `graphspec/schema.py:5` | 2.3, 2.10 to 2.13 |
| Splits are train, validation and spread | `tuning/build.py:20` | 2.3 |
| One feature vector of one width for every kind of node; values on relations are numbers | `docs/contract.md:26`, `:31` | 2.1 |
| Relations are declared kinds, few in number | `graphspec/registry.py` | 2.1 |
| An edge joins two nodes; an edge cannot be the endpoint of another; a node cannot hold a graph | whole core | 2.4 |
| Variants come from seven stages in a fixed order | `graphspec/transform.py:15-23` | 2.6 |
| Representations are a closed list | `graphspec/representations.py` | 2.9 |
| A graph fits in memory whole | `graphspec/resources.py:10-11` | 6.4 |
| A fit needs a GPU | `tuning/training.py:128` | 1.5 |

Missing from the library:

| Missing | Nearest thing today |
|---|---|
| A builder that takes sources, keys, references and time | `GraphTables`: nodes, observations, edges, node targets, integer steps, one graph; used by one adapter |
| Conversion of a database from the keys and views it declares | none |
| Operations to slice, continue, fork, edit, merge on keys and compare | `tuning/offers.py`, which does the first four by building the engine's internal objects by hand |
| A declaration of decisions and options, and measures of how well a model tells options apart | one score, fixed in code |
| One call to derive a variant; one to validate staged tables | internal functions |
| Prediction without a training session; a search started from Python | `TrainingSession`; the CLI |
| Operations, targets, column types and readers registered by users | closed lists in code |
| Sampled batches from one large graph | none |
| One record holding every choice of a fit or a search | `plan.json`, flags of the command line, constants |
| A description of what a graph holds after each operation, of what of an option is connected to a task, and of what a model reads and responds to | none; the questions were answered on 28 September by reading code and by scripts written for the occasion |

Structure:

| Finding | Where |
|---|---|
| `graphspec` imports `tuning` and `tuning` imports `graphspec` | `graphspec/api.py:11-40` |
| Four routes build a derived graph in production; they share the stage operators but not the orchestration or the assembly of representations | `segments.build_parallel`, `transform.materialize`, `segments.build`, `offers.Line` |
| Four writers of staged tables | `graphspec/staging.py`, `vic3/tables.py`, `totalwar/tables.py`, `graphspec/synthetic.py` |
| Two records named failure policy, with different content, one per kind of selector | `tuning/space.py:22`, `tuning/selector.py:20` |
| Two formats for the scaling of targets | `tuning/data.py:141-148` |
| 34 modules under `graphspec/` and `tuning/` define functions and contain no timing | among them `graphspec/operators.py`, `graphspec/kernels.py` |
| Importing the library requires `GRAPHSPEC_HOME`, creates a directory and sets 16 environment variables | `graphspec/infra/runtime.py:17-64` |
| Memory limits of graphs are constants that no setting reaches | `graphspec/resources.py:10-11` |
| A bank without two varying validation targets is refused | `tuning/build.py:142` |

Results that depend on order, machine or chance:

| Finding | Where |
|---|---|
| The selector of each trial is drawn without a seed | `tuning/selector.py:260` |
| The candidate set depends on the number of cores | `tuning/space.py:154` |
| Dense algebra moves to the GPU at size 1,024 without checking that one exists | `graphspec/infra/compute.py:17` |
| One family seeds its random basis with the trial's seed plus the graph's index, on a generator of the batch's device | `tuning/events.py:396-399` |
| Sampling and dropping hash an entity's position, with one constant seed for every graph | `graphspec/operators.py:403`, `:1125`, `:1157` |
| Rewiring is seeded by step number | `graphspec/operators.py:890` |
| The key of the variant cache covers every `.py` file of `graphspec/` and none of `tuning/` | `graphspec/transform.py:13`, `tuning/data.py:323` |
| The bank stores features at 16 bits | `graphspec/truth.py:238` |

Defects in the uncommitted code of 28 September:

| Defect | Where |
|---|---|
| Forks share counts that are changed in place, so the options of one decision affect each other and later decisions | `tuning/offers.py:203-212`, `graphspec/representations.py:92-93` |
| Graphs holding options are found by parsing names, and separately by having no train or validation rows | `tuning/offers.py:278-288`, `tuning/data.py:132-133` |
| The one score of how well a model tells options apart rewards movement whether or not it is right, grows with the number of options, and is set by the single most unusual option | `tuning/offers.py:443-455` |
| Nothing stops a search from removing every route by which an option reaches the model | `graphspec/spec.py`, `tuning/selector.py` |
| Constants in days; three series assumed; wording of one dataset | `tuning/references.py:8-9`, `tuning/catalog.py:58` |
| No test covers slicing, replacing a step, forking, or prediction over options | `tests/` |

What the running search showed on 28 September at 18:10, as evidence for sections 3, 4, 8 and 9: validation losses of 23 finished trials lay between 0.250 and 0.277 while the term for spread ranged from 0.33 to 1.0, so the order of trials was the order of their spread; every leading trial had dropped the nodes of actions; in the best trial the recorded option was the highest or lowest of 8 in 82% of 60 decisions, against 25% by chance, and half of the spread came from 5 decisions; two families ordered the same options in opposite directions. The score rewarded telling the recorded option from regenerated ones.

What answering questions about that search took on 28 September, as evidence for sections 4 and 9:

| Question | How it was answered | Outcome |
|---|---|---|
| What of an option does the best design read | A reading of seven modules | Wrong; corrected after a script compared stored predictions of options alike in one respect |
| Are the nodes of actions connected to anything at the time of choosing | The same reading, then a rule applied by hand to 41 configurations | One setting, the number of consecutive times an edge must last, disconnects them in 22 of 23 trials that keep their edges; nothing reported it |
| Does the design predict better than predicting no change | A script over stored predictions | No, on two of three series; the search's records did not show it |
| Why is the spread of the best design the highest | A script over stored predictions | In one trial the recorded option was the extreme of 8 in 82% of decisions; half the spread came from 5 of 60 decisions |
| Which operation removed the edges of actions | A reading of the code | Not confirmed by any run |

Faults in the adapters that a rebuild must not carry over:

| Fault | Where |
|---|---|
| Whether a proposal was accepted and whether it succeeded are values of a fact | `totalwar/events.py:80-81` |
| The result of an event is a feature of the query it belongs to, and a fact at the same step | `totalwar/actions.py:51`, `:54-57` |
| A field named `answer` is a query feature; what it means is recorded nowhere in the repository | `totalwar/actions.py:38` |
| Every missing value on an edge becomes 0.0 | `totalwar/values.py:44-47` |
| Tables merge onto a node without prefixes, so columns of the same name overwrite each other | `totalwar/state.py`, `totalwar/world.py` |
| Events receive a removal at the next step as if they were states | `totalwar/graph.py:73-93` |
| Parallel edges collapse; the last row wins | `vic3/graph.py:321` |
| A feature tells regenerated options from recorded ones | `vic3/redraw.py:21`, `vic3/graph.py:776` |
| All content of an option passes through one node | `vic3/graph.py:768-773` |
| Regenerated options do not record the probability under which they were drawn | `vic3/redraw.py` |
| One adapter states no intervals: one graph at one step | `churn/truth.py:95` |

**What outside practice adds**

Read on 2026-09-28. Each finding names the subsection that uses it.

| Finding | Sources | Used in |
|---|---|---|
| Tools on the learning side type columns and define tasks and splits but assume node and edge tables exist; tools on the data side map and reshape but know nothing of tasks or leakage | RelBench, PyTorch Frame, PyG, DGL, GraphStorm; R2RML, SQL property graphs, Neo4j, TigerGraph | section 5 |
| A table is its key, its references and its time column; an interval is two columns; one table may be both nodes and edges; a task is a separate table of entities, a time and a target; a graph can be declared as a view over the tables of a database | RelBench, Kumo, SQL/PGQ; two lectures of Stanford CS224W | 2.3, 5.7, 5.8 |
| Time has several models in use: none, order only, snapshots, a regular signal on a fixed graph, a stream of events, intervals, and two clocks (when true, when known) | surveys of Skarding and of Longa; TGB; Raphtory; PyG Temporal; tsl | 2.2 |
| Snapshots of many relational tables are served by almost no tool | same | 2.2 keeps them |
| An operation takes a graph and returns a graph; a pipeline is a list the user writes; users register operations beside built-in ones | PyG and DGL transforms, NetworkX views, GraphGym registries | 2.5, 2.6 |
| Path composition, projection, multi-hop edges and diffusion are products of relation matrices; rewiring is taught as a standard operation | GraphBLAS, DGL; Cambridge L65 | 2.9 |
| No library forks at a time and applies other content; the known mechanism is a snapshot with a private overlay, the shared history computed once | Aspen, DeltaGraph, Raphtory | 2.7 |
| Construction of the graph changes results as much as the model | 4DBInfer, AutoG | 2.6 keeps construction in the search |
| Guards against leakage: nothing sampled is later than the task's time; the end of an interval is not visible before it happens; statistics come from training rows; ties in time are ordered by rule; edges used as targets are kept out of the messages | RelBench, TGB, Raphtory, PyG | 2.1, 2.2, 2.3, 2.10, 5.10 |
| Better tools stop when a value is missing and no rule was stated | Neptune ML, Neo4j import | 5.4 |
| Writing back the resolved configuration with fitted vocabularies, and the map from keys to ids, lets prediction repeat the transforms | GraphStorm | 5.2 |
| One large graph needs neighbourhood sampling that respects time, storage that is not loaded whole, and partitioning | PyG loaders, GraphBolt, TGL; engineering reports of Uber and Pinterest | section 6 |
| Tasks beyond regression: classes, several classes, ranking, links with sampled negatives, proportion vectors, sequences, time to an event | OGB, TGB, RelBench | 2.10 to 2.13 |
| An option is a row with features, references to nodes by role, parameters, a mask, the decision it belongs to, and the probability under which it was shown or drawn; no format in use holds all of these | AlphaStar, entity-gym, SlateQ, learned branching, Vowpal Wabbit, Open Bandit | 3.1 |
| Systems that score options with targets read the embeddings of the targets and a summary of the state; none passes an option through one node | learned branching, S2V-DQN, L2D, OpenAI Five, MINERVA | 3.4 |
| Models that ignore the action are a known failure; the remedies are structural | MP-DQN, C-SWM, SIN, R-learner, dueling networks | 8.8 |
| Validation loss predicts poorly which model chooses better; differences between options are small in good models too, so order matters more than size | Lambert et al., dueling networks, Farahmand | 8.3 to 8.6 |
| Measures that need only the outcome of the chosen option exist; measures weighted by the probability of the logged choice exist where that probability is known | Genie, permutation importance, R-loss, uplift; inverse propensity, doubly robust, slate estimators, fitted Q evaluation | 8.4 to 8.6 |
| Options drawn afterwards carry the bias of their generator; a classifier that tells drawn from logged options better than chance shows it; the correction for sampled options needs the probability of each draw | Sampled MuZero, sampled softmax, classifier two-sample tests | 3.2, 8.9 |
| Scoring K options with shared history is exact when options are batched beside the shared graph and messages pass from state to options only | incremental inference, late interaction | 3.5 |

**Scope**

In scope: graphs built from tables, databases that declare their keys, files, archives, nested files and event logs; every time model in the table above; nodes; nodes that hold a graph; edges between two nodes with values, parallel, signed, weighted, directed or not; edges joining a set of nodes; edges whose endpoint is an edge; relations given by a column as well as declared; edges computed from a function of two rows' columns, such as distance, similarity or nearest neighbours; tasks on nodes, edges, sets of nodes, ordered paths, subgraphs and graphs; decisions among options, both where the options were recorded and where only the chosen one was and the others are regenerated, with one actor or several; registered kinds of target; many small graphs and one large graph on one machine with one GPU; prediction on graphs and entities not seen in training, by the families that state it.

Out of scope: several GPUs or several machines; simulation of physics and models that depend on rotation in space; extracting a graph from free text with a language model; models that generate graphs. Text columns are carried as categories or as vectors the user supplies. Records that share no key are joined through a table of matches the user supplies.

**Where the old plan and this plan differ**

| Subject | Old plan | This plan | Reason |
|---|---|---|---|
| Builder | One call per fact (1.1) | Operations over source tables and mappings, which produce the same facts; the fact calls stay as the base layer (5.1) | With 73 source tables, fact calls alone leave one hand-written loop per table |
| Names of the fact calls | `node`, `edge`, `attributes`, `label`, `query` (1.1) | `node`, `edge`, `event`, `change`, `task` (5.1) | Queries and labels are one thing; events had no call |
| What stays in a script | Reference tables, catalogue of actions and rule of reward (2.2) | Names, source statements the library cannot read from the source itself, declarations, and functions where no declaration serves, each recorded | The old acceptance test already said anything beyond reading the source and stating facts is a missing feature |
| Reading a database | The script uses the driver directly (2.2) | A reader of the library, under one extra (5.3), and conversion from the keys the database declares (5.8) | Every script with a database would repeat it |
| Rule for targets | Written by the user as logic (acceptance test, 2.1) | Declared as an expression over later rows; a function remains allowed (5.10) | The rules seen are one operation with different settings |
| Features | Hashed from dicts by value type into one vector of a default width (1.2) | Columns have stated types; width is per kind (2.1, 5.4) | One width for all kinds is an assumption of the core |
| Default split | By key hash or last periods when not given (1.2) | A split is stated (5.11) | A default is a fallback |
| Mechanics of time | Steps from distinct times, restating, removal rows (1.2) | Facts carry when they hold and when they were known; steps are a view (2.2) | Restating and removals are one way of storing intervals |
| Queries | Three edges per query (1.2) | Tasks with named references (2.3) | Three roles are an assumption of the core |
| Errors | At call time, by named rule (1.3) | Same rules; all errors of a build are collected and reported together (5.13) | One run shows every problem |
| `summary()` | A method beside the verb `inspect` (1.4) | `inspect`, as a verb and as a call from Python on a builder | One name for one thing |
| Layout of files | `build --data DIR` reads a fixed layout (1.6) | The same layout, read as a source with a mapping shipped with the library (5.3) | One mechanism |
| Public surface | Six names and seven verbs; `GraphTables` and `write_task` removed; nothing outside the library imports `graphspec.infra` (1.7) | 8.2; the same removals and the same rule; `configuration` becomes `statement`; the verb `database` is removed | Operations, prediction and search from Python were absent |
| Documents | Rewritten in section 1 (1.8) | Rewritten in the subsection that changes what they describe | A document must not be wrong for the length of the plan |
| Check of rebuilt datasets | Fingerprints compared, allowed to differ (2.3) | Content compared by the library's comparison of graphs, against a list of intended differences (7.5) | A fingerprint says nothing about what differs |
| Existing banks and searches | Stay valid (2.3) | Banks of version 7 are refused by the code of section 2; their searches stay readable by the dashboard, and are continued only from the source kept in 1.1 | The core model changes what a bank holds |
| Coverage of `synthetic` | Intervals, changes, queries, signals, each level (2.5) | Every row of the table of assumptions (acceptance 4) | The old list covered what the two datasets used |
| Where scripts live | Under the runtime home (2.4) | In `local/` of the checkout, which git ignores | Owner's instruction of 28 September |
| When the adapters leave | After the scripts exist (2.4) | In 1.4, once their staged tables and source are kept | The core beneath them changes in section 2 |
| Stored names | Two names of scaling stay; one name of time renamed in memory only (4.2) | Renamed in bank version 8 (2.1) | Path banks do not store the first two, the third is a stored key, and section 2 needs a new version |
| Acceptance | Ends with `search run` | Ends with one fit inside the test suite; searches on real data are section 11 | A search does not fit a test of 20 seconds |
| Order | Builder, datasets, search, cleanup | Ground, core, decisions, descriptions of graphs, builder, scale, datasets, tuning, descriptions of models, search, runs on real data, cleanup | Scale, decisions and descriptions are new |
| Verbs refused in a checkout | `build`, `check`, `fit`, `search` (3.1) | `build`, `check`, `search`, `benchmark` (10.1) | `fit` is not a verb |
| The record of a search | `plan.json` keeps version 9 and its shape; six settings in `settings.json` (3.3) | `plan.json` becomes version 10 and holds the statement of 1.6; `settings.json` holds the fields of it that may change between trials | The choices a run must state grew, and had no one place |
| Detached start | One process started inside `search run` (3.4), replacing a scheduled task | Same; it replaces `scripts/detached.ps1`, added on 28 September | The scheduled task no longer exists |
| Timing files | Two deleted (4.1) | Three deleted (12.1) | A third was added since |
| Seeds of the search | Not mentioned | The selector draw is seeded and the candidate set no longer depends on the machine (8.10) | Two searches of one statement on one machine differ today |

Decisions of the owner that this plan keeps as they are: no cross-validation; options whose outcome is not known never enter a loss as targets; path targets are scored without skill or R²; a search starts with 10 random trials and then mixes random, TPE, forest and BoTorch equally; trials are budgeted in seconds.

Not settled by the owner, and therefore a parameter of the library and not a rule of it: how the ability of a model to tell options apart is measured, and how that enters the choice of a model. The score of the search running on 28 September (validation loss minus 0.05 times the base-two logarithm of spread, spread floored at 1e-6 inside the score) stays available, unchanged, as one measure and one way of combining (8.3, 8.4).

**Acceptance test**

1. The walkthrough in `docs/loader.md` runs as a test: a script of declarations, then `inspect`, `build`, and one fit. It contains no loop over rows, numbers nothing and builds no table. Checked in 5.15.
2. The three datasets that were adapters are rebuilt as scripts in `local/`. For each, the status report gives its statements split into names, source statements, declarations and functions, and lists every function with the reason no declaration serves. Reported in 7.5.
3. A test fails if the name of a dataset that was an adapter, in any case or spelled out, appears in `graphspec/`, `tuning/`, `tests/`, `docs/`, `README.md` or `pyproject.toml`. The test holds the names split, and excludes itself. Written in 1.7, extended in 7.7.
4. For every row of the table of assumptions fixed in the core, a synthetic case that violates it builds, validates and completes one fit on CPU inside the test suite. Each case is the check of the subsection named in that row.
5. Every dataset named in the column Use of the table of public data builds and validates from a script in `local/` within the budgets of section 6, and one fit of 1,000 steps completes on each. Every function in those scripts is recorded as a defect of the library and removed by a declaration. Checked in 7.4 and 10.2.
6. Five further accounts of graphs to build, written by a reviewing agent that has not read section 5, two of them with decisions, are built. Each change to the library they need is recorded as a defect and made. Checked in 5.15.
7. Every question of the list of 9.7 is answered by a verb or a call of the library on a synthetic search, and a test compares each answer with the known truth of the synthetic data. No answer is obtained by reading code. Checked in 9.7.

---

**1. Ground**

Done before anything in the library changes.

1.1 The working tree is copied whole to the runtime home. Searches on banks of version 7 are continued only from that copy. Check: the copy's fingerprint equals that of the working tree.

1.2 From that copy, started detached: the staged tables of the three adapters are built and kept under the runtime home; for the synthetic source and ten configurations listed in this plan when the fixture is made, which between them use every stage and every representation, the staged tables and the arrays after each of the seven stages and after the representation, under today's code, are stored as a fixture. Check: both exist with their fingerprints recorded in this plan.

1.3 Under `graphspec/`, `tuning/`, `tests/`, `docs/` and the adapter packages the working tree is set to `main`: modified files are restored and the untracked files of 28 September are deleted, the copy of 1.1 holding them. `PLAN.md` is not restored. Two corrections of 28 September are then applied, each with its test: the guard in `tuning/inputs.py` against reading the last entity's token for a node that has no entity, and the single thread for dense algebra in `graphspec/infra/compute.py`, which makes derivation independent of the number of threads. Check: under `graphspec/`, `tuning/`, `tests/`, `docs/` and the adapter packages, `git status` shows only those two files and their tests, and the suite passes.

1.4 The three adapter packages, their tests, verbs, documents and packaging, `graphspec/infra/postgres.py`, and their names in `tests/test_library.py`, `tuning/README.md`, `README.md` and `docs/loader.md` are removed. `local/` is added to `.gitignore`. Check: the suite passes.

1.5 A fit takes its device as a required argument, runs on CPU when it says so, and stops with an error when it says GPU and none exists. 1.6 moves the argument into the statement. Check: a synthetic fit completes on CPU.

1.6 The statement of a fit and of a search begins. One versioned record; every field it holds is required, and an absent field stops the run with an error naming it. It starts with device, seeds and budgets. Each later subsection that says the statement holds something adds its field and the test of its absence. Check: a fit with each field absent in turn stops naming the field.

1.7 Standing tests. The test of timings: a module under `graphspec/` or `tuning/` that defines a function and contains none of `perf_counter`, `timed`, `measure(` fails it; every module the test names is given timings. The test of names of acceptance 3, over `graphspec/` and `tuning/`. A test that fails if `graphspec` imports `tuning`, or if anything under `graphspec/` or `tuning/` reads `local/`; for it the four functions of `graphspec/api.py` and the command `graphspec` move to `tuning`, and `scripts/detached.ps1` and `scripts/graphspec.ps1` call the command where it now is. Check: the three tests pass.

**2. Core model**

Each check of this section ends the same way unless it says otherwise: the synthetic case builds, validates, reads back from the bank equal to what was stated, and completes one fit on CPU.

2.1 Schema and bank version 8. One object states the kinds of node, the relations with their endpoint kinds and whether they have a direction, the columns of each with their type, the attributes of the graph itself, and the unit of time. A relation's kind is declared or taken from a column. Each kind of node has its own columns and width. How each type of column becomes input to a model is an encoder, registered, fitted on the rows of the splits trained on. The schema is versioned, printed by `inspect`, and checked after an operation only for what the operation states it changes. Bank version 8 starts here and grows through 2.2 to 2.4, 2.10 to 2.13 and 4.1. No name of a level of grouping is stored; groupings are columns stated by the script. Stored names change: the two names of scaling of a single target become the `series` list; the row keys that name a level of time and a query become the columns the script states; the array that names the joined graph of a step keeps its content under the name `step_graph`; the three arrays of roles become named references. Banks of version 7 are refused with an error naming their version. A reader of the fixture of 1.2 gives its tables and arrays under the names of version 8; it is the only code that reads the old form, and equality with the fixture means equality of these after sorting by key. `docs/contract.md` is rewritten here and again in each subsection that changes the bank. Check: a synthetic graph with two kinds of different width, a relation with a category as value, a relation without direction, and relations taken from a column.

2.2 Time. Every fact carries the interval in which it holds. Where it differs from the start of that interval, it also carries the time from which it was known, and the time from which it was replaced by a correction. A fact may state who can see it; a task may state who is looking. Shorthand forms, registered: a state over an interval, an event at one time, a snapshot of a table at a time, a change of a value at a time, a signal at regular times on nodes that do not change. A graph may have no time, order without a clock, or two clocks. Graphs of one bank may have different units. Facts with the same time are ordered by a stated rule. Several facts about one entity at one time are allowed. Steps, periods and windows are views chosen by the pipeline. Check: one synthetic case per form, each equal to the same content stated as snapshots; a fact known later than it holds is invisible to a task between the two times; a corrected fact is seen in its old value by a task before the correction; a fact visible to one observer is invisible to a task of another at the same time; a case with three facts about one entity at one time; a case without time, a case with order only, and a bank of two graphs with different units, in none of which a period is stated.

2.3 Tasks. Queries and targets become one table of tasks. A row has a kind, a time, named references, optional features, an optional target with the time from which the target is known, an optional weight, and a split. The sorts of reference are registered; shipped: a node, an edge, a set of nodes, an ordered path of edges, a subgraph, the graph. Names and number of references are stated per kind of task. Several tasks may share a time. A split is any name. For each split the statement of the fit says which facts its tasks may see, so that edges used as targets are kept out of the messages and entities of a scoring split can be hidden in training. A target whose known time has passed may be read as a fact by later tasks. Models receive references as a set keyed by name. Whether an operation may remove a referenced node is stated by the operation. Check: synthetic cases with no task at some times, several at one time, kinds with one and with five references, a reference of each sort, and a task that reads the matured target of an earlier one, a bank whose splits are four names, none of them train, validation or spread, of which the statement trains on two and scores on two, and an operation that states it may remove a referenced node and does; a reference to a node not observed at the task's time; and a bank with targets on nodes and on graphs together.

2.4 Edges beyond pairs, and nodes that hold graphs. An edge joining a set of nodes is a kind of its own, with a role and values per member and features of its own. An edge may be the endpoint of another edge. A node may hold a graph of its own, encoded by a stated encoder. Forms of storage for edges joining sets are operations; shipped: a node with one edge per member, one edge per pair of members. Operations that count degree, components or paths state how they treat them. Check: synthetic cases with sets of two to twenty members, with an edge about an edge that carries its own time, and with nodes that hold graphs of five nodes.

2.5 Operations. An operation takes graphs and parameters and returns a graph. Its input is never changed; unchanged parts are shared, not copied. It states its parameters with their ranges, which splits it may read, what it requires of the schema and what it changes in it, and its version. Operations are registered by name under a namespace; a name registered twice stops with an error. The seven stages become registered operations. Check, without a fit: an operation registered by a test is listed with its parameters; each of the seven, called alone on the fixture of 1.2, gives the arrays of that stage.

2.6 Pipelines and the key of the cache. A variant is a pipeline. A fit states its pipeline. The search covers which operations of the pipeline are present and their parameters, not their order. The key of the variant cache is made of the schema, the pipeline with the versions of its operations, the seeds and the fingerprint of the source, and hashes no file. Check, without a fit: the seven operations in today's order give arrays equal to the fixture of 1.2, those of sampling, dropping and rewiring until 2.8 replaces their part of the fixture; changing the version of an operation changes the key; editing a file without changing a version does not. With a fit: a pipeline of the seven in another order, and one with an operation twice, each builds and completes one fit on CPU.

2.7 One engine, forks and edits. Derivation of a whole graph is the segment engine with one segment. With one worker it runs in the calling process. `transform.materialize`, `segments.build` and `segments.derived` are removed. The physical join stays as what tests compare against. Slicing by time, continuing a derivation, forking and applying an edit are built here as operations. A fork is the state at a time plus an overlay; the history is derived once and shared. Check, without a fit and with a budget of 2 seconds: for every registered operation and representation, a fork, its sibling forks and the trunk equal the whole graphs derived separately; arrays are equal for 1 and 2 workers; for N edits at one time, the time is at most that of one derivation of the history plus N derivations of the last step plus 10%. Equality for 8 workers is checked in 10.1.

2.8 Identity in random draws. Every draw is keyed by the hash of the entity's key and of the graph's name, never by position, index, step number or device. The part of the fixture of 1.2 that holds sampled, dropped and rewired arrays is replaced here. Check, without a fit: permuting entities and graphs leaves every derived array equal after sorting by key; the arrays are equal on CPU and on GPU.

2.9 Operations added. Each row is one unit of work with one test on a synthetic graph, without a fit; the subsection is done when all are. Representations are registered in three families: those that change structure, those that sample, those that encode; a representation registered by a test completes one fit on CPU. Each operation added passes the check of forks of 3.7.

| Operation | The concept it serves |
|---|---|
| Merge of graphs on keys | Algebra between graphs |
| Difference of two graphs | Algebra between graphs |
| Edges from a function of two rows' columns: distance with a threshold, nearest neighbours | Structure that is computed, not given |
| Edges between consecutive rows of a group; groups cut by a gap in time | Order as structure |
| Neighbourhood of a node to a number of hops, at a time | Working on part of a large graph |
| Parallel edges kept, or reduced by a stated aggregate; reverse edges; self loops | Forms of one relation |
| Product of relations, and its closure; path composition, projection, multi-hop edges and diffusion are rebuilt on it | Relations derived from relations |
| Contraction of groups of nodes | Coarsening |
| Degree, components and core numbers as features | Measures of structure |
| Negative sampling of links and of sets, by a stated rule | Tasks about what is absent |

2.10 Targets: registry, numbers and paths. Kinds of target are registered. Each kind of task states a utility. Losses and measures are registered separately from targets, and the statement of a fit says which it uses. Scaling is stored per series for every numeric target. Shipped here: a number; a path of numbers over stated series and offsets. Check: one synthetic case per kind; a kind registered by a test.

2.11 Targets: a class; a set of classes; a proportion vector. Check: one synthetic case per kind.

2.12 Targets: a ranking among listed items; a link among sampled negatives; a set of nodes. Check: one synthetic case per kind.

2.13 Targets: a sequence of tokens; the time to an event, with the cases that ended before it happened. Check: one synthetic case per kind.

**3. Decisions and options**

Working with graphs in which an actor chooses among options is a key capability. What is asked of a model there is to order the options well, more than to predict outcomes closely. Both cases are served: the options that were offered are recorded; or only the chosen one is, and the others are regenerated.

3.1 The declaration. A decision is a task with options. It states its actors, one or several; the decisions of other actors taken at the same time; and, where decisions follow one another, the next decision of the same actor, the end of an episode and a discount. An option holds one action or several, and states whether they are a set or a sequence. An action holds a kind, an actor, targets with their roles (a node, an edge, or an element the action creates), and parameters with their types; numbers stay numbers. The kind of task states which option, if any, is doing nothing. Each option also holds: whether it was available; whether it was chosen, and its order where several were; the probability under which it was shown, if known; the probability under which it was chosen by the policy that logged it, if known; where it comes from (recorded, or regenerated by a named generator with its seed and the probability of the draw, if known); and its outcome, if known, per option, per action or per position. Options may be listed, or given by a rule where they are too many to list, and then sampled by 4.2. An option without an outcome never enters a loss as a target. Whether an option was available is never a model feature where it records that an action validated. The declaration says nothing of how an option is realised. Check, as in section 2, the fit reading the task and the outcome of the chosen option only: synthetic decisions with one action and with several, as set and as sequence, with targets that are nodes, edges and new elements, with two actors at once, with outcomes on every option and on the chosen one only, recorded and regenerated; a test that no option without an outcome reaches a loss as a target.

3.2 Generators of options. Registered; shipped: a function of the script; the options of other decisions of the same graph whose targets resolve; the options of the nearest states; changes to the chosen option within a stated range; a sample from a rule of 4.1. A generator states the probability of each draw where it has one; where it has none the manifest records that. Check, without a fit: each shipped generator on synthetic decisions gives options whose targets all resolve.

3.3 Realisations: registry and guards. How an option is put before a model is registered, and is stated by the fit or searched. Messages pass from state to options only, unless a realisation states that they pass back. Before every fit the library checks: a route exists from each field of an option's content (kind, targets, parameters) to the prediction; with weights drawn by a stated seed, the derivative of the prediction with respect to the encoding of each field is not zero for at least one option of the scoring split; the score of an option does not depend on the content of another unless the model states that it reads them together; a set of actions gives the same encoding in any order. The share of pairs of options of different content with equal predictions is recorded, and the statement of the fit says the share above which the fit stops. Check: a configuration with no route from options to the prediction is refused before the fit with an error that names the field; permuting a set of actions leaves the encoding equal.

3.4 Realisations shipped. Each row is one unit of work with one check; the subsection is done when all are.

| Realisation | Check |
|---|---|
| The option is an existing node of the state | Equal to scoring each option alone |
| A vector built from the embeddings of its targets by role, its kind and its parameters, scored against a summary of the state | Equal to scoring each option alone |
| Nodes for its actions joined to their targets by role | Equal to encoding the K graphs separately |
| An edge between actor and target | Equal to encoding the K graphs separately |
| An edge joining actor and targets as a set | Equal to encoding the K graphs separately |
| An edit of the graph, encoded afterwards, with messages passing back | Equal to encoding the K graphs separately |

3.5 Cost of scoring K options. All options of one decision read one encoding of the history. The depth to which state and option interact is a parameter, zero meaning that each option is scored from the encoding of the state alone. Check, with a budget of 2 seconds: for each realisation whose messages pass from state to options only, the time for K options is at most that of one encoding of the history plus K of the last step plus 10%, for K of 2, 8 and 50.

**4. Descriptions of graphs**

What a graph holds, what a pipeline did to it, and what of it is connected to a task are written by the library from the data. Every description is a file beside what it describes, in words and tables for a person and in a form a program reads, and is printed by the verb `describe` and by a call of the same name.

4.1 Description of a graph. For a truth graph: per kind of node, the number of nodes and of observations, over all times and per time; per relation, the number of edges, the kinds at its ends, the columns of its values, how many hold at each time and for how long; the time model and its range; per kind of task and per split, the number of tasks; per decision, the number of options, how many are recorded and regenerated, and the generators. Check, without a fit: on a synthetic graph every number of the description equals the number written out in the test.

4.2 Description of a pipeline's effect. For a variant, per operation in the order applied: what it removed and what it added, per kind and per relation, as counts and as shares of what was there; which settings of the operation were in force; and the state after it. An operation that removes every node of a kind, or every edge of a relation, at the times of the tasks is named as such. Check, without a fit: on a synthetic pipeline whose second operation removes every edge of one relation at the times of the tasks, the description names that operation, that relation and those times.

4.3 Reach of a task. For a task and for each of its options: which fields of content (kinds, targets by role, parameters, features) are connected to the task's references at the task's time, by which relations and at what number of hops; which are present in the graph and connected to nothing; and which are absent. For what is not connected, the description names the operation and the setting that cut it. Reach is computed from the variant and the realisation, never from the code of a model. Check, without a fit: on synthetic decisions with a pipeline in which one setting removes the edges of options at their first time, the reach of every option names that setting; with the setting changed, the reach lists the fields of the options at one hop.

4.4 Reach over a search. For a set of variants, one table with a row per variant and the answers of 4.2 and 4.3 as columns: for each field of an option's content, whether it is connected at the time of choosing. Check, without a fit: over twenty synthetic variants the table equals the one written out in the test.

4.5 Differences. `describe` of two graphs or two variants gives what one holds that the other does not, per kind and relation and per time, using the difference of 2.9. Check, without a fit: two synthetic variants that differ by one setting differ in the description by what that setting removes and nothing else.

**5. The graph builder**

5.0 The language first. The syntax of declarations is fixed in `docs/loader.md`. A statement holds one declaration and is one top-level statement after the imports; the types of the columns of one source are one statement. The descriptions of 5.15 are written in it as scripts, within the ceilings of that table. Done when the scripts are in the repository; they cannot run yet. A script that later has to change is recorded as a defect of the language, with the change.

5.1 The base layer. One call per fact: `node`, `edge`, `event`, `change`, `task`. Check: a synthetic graph stated by calls equals the fixture of 1.2 for that graph.

5.2 Output. Writes in chunks within the budget of memory of the statement; shards in parallel and their merge; reuse of staged output; appending later times to staged output and to a bank without rebuilding what came before; a manifest with a fingerprint of the sources, the resolved declarations with fitted vocabularies, the map from keys to ids, and, from 5.9 on, the counts of 5.9. Kinds and relations are ordered by name. `inspect`, as a verb and as a call on a builder, prints the schema, the time model, counts, tasks per kind and split, and skipped references. Check: a build in three shards equals the build in one; appending the second half of a synthetic source to a build of the first half equals the build of the whole; two builds have equal fingerprints.

5.3 Sources. A source is anything that yields batches of records with a schema. Readers are registered by name. Each row is one unit of work; its check is that the reader yields the same records from the same synthetic content.

| Readers |
|---|
| Parquet; CSV and delimited text without a header |
| Postgres; SQLite; keys read from the database where it declares them |
| JSON and nested records; HDF5; NumPy archives; pickled arrays |
| Archives of several members, read by member without unpacking to disk; compressed forms of each reader |
| A directory in the fixed layout of files (`nodes/<kind>`, `edges/<relation>`, `tasks`), with a mapping shipped with the library; the task file is removed |

Database drivers are dependencies under one extra. A reader registered by a test is used by a script.

5.4 Columns. Types of column are registered. Shipped: number, ordered category, category, flag, time, duration, interval, list, vector, point, identifier, nested record, text. `inspect` proposes types; the script states them; a column of unstated type stops the build. Values that mean missing are stated per column. A column with missing values needs a stated rule; rules are registered; shipped: omit, stop, a value. Check: an unstated column, and a missing value without a rule, each stop the build with an error naming the column.

5.5 Expressions. One language for computed columns, keys, predicates and targets: arithmetic, comparison, text patterns, parts of dates, bins, conditionals, lookups, a literal for missing, and aggregates over a whole table used as single values. Its functions are registered. Check: each shipped function on a synthetic column equals the values written out in the test.

5.6 Operations on tables, registered. Shipped: select rows, compute columns, join (on keys, on inequality, on a window of time), group with aggregates, pivot, unpivot, expand a list or a count into rows, expand nested records to any depth, cut a flat column into groups by a column of counts, align files by row number, union, order within a group. Check: each against a small table written out in the test.

5.7 Mappings, registered. Shipped: rows to nodes; rows to edges; rows to edges joining sets; rows to tasks; rows to decisions, options and actions. Each mapping states its key, its times, and its columns. Identity: single and composite keys, keys within a namespace, keys of edges, keys made from the position in a nested record, keys from the row number of a file, nodes from the distinct values of a column, nodes created on first reference only when stated. Mappings produce the calls of 5.1. Check: one synthetic case per mapping and per rule of identity, each equal to the same graph stated by calls.

5.8 A database that declares its own structure. The builder reads from the catalogue of the database, or from a file of definitions in SQL, its tables, primary keys, foreign keys, views and column types. A table with its own key becomes a kind of node; a foreign key becomes an edge; a table whose key is made only of foreign keys becomes an edge with its other columns as values; a view is a source. Each of these rules can be overridden for a named table by one statement. The script supplies four things, each as a function or a name: the target; the columns of time of each table, or a rule that finds them by type; which tables are read; and eligibility, a function of the time and of the main key that says which entities get a task and which rows a task may see. Check: a synthetic database of twelve tables with declared keys, two link tables and two views converts in six statements, and the same six with another target and another eligibility convert it again.

5.9 Presence. A reference to a row absent at that time follows a stated rule; rules are registered; shipped: stop, count and skip. Counts appear in the manifest. For a table seen only at some times, the rule for the times between is registered; shipped: absent, and carried forward with its age as a column. Check: each rule on a synthetic table gives the rows and counts written out in the test; an unstated rule stops the build.

5.10 Tasks. Read from rows, or made by a rule such as every node at every time or every event. A target is an expression over rows later than the task's time; a function is allowed and is counted under acceptance 2. The builder refuses a feature or fact known later than the task's time. A column stated as an outcome is refused as a feature, as a fact before the time from which it is known, in a filter and in a split. Check: each refusal raises an error naming the column.

5.11 Splits. An expression or a column; forms are registered; shipped: a hash of a key with a fraction and a salt, a time, a group, a graph, a split that comes with the data. One hash function. Check: each form on a synthetic table gives the members written out in the test; two builds give the same members.

5.12 Decisions and options are mapped from tables of decisions, options and actions, or from a generator of 3.2 named by the script. Check: both routes give the options of the synthetic cases of 4.1.

5.13 Validation collects every error of a build before reporting, names the declaration and the source row, and never names an internal table. Rules local to one call are checked at the call. Check: a script with three errors of different kinds reports all three, each with its declaration.

5.14 Verification against the source is made from the declarations: at a stated number of sampled times, the declared tables are read again and compared with the staged graph. It is how a user knows a build holds what the source holds. Check: a synthetic source changed after the build is reported with the table and row that differ.

5.15 Check of the section. `synthetic` becomes a script of declarations shipped with the library. The walkthrough of `docs/loader.md` runs as a test. The scripts of 5.0 build, validate and fit in the test suite from synthetic sources, each within its ceiling of statements, with a budget of 0.5 seconds each. A script over its ceiling is a defect recorded against the missing handler. The five further descriptions of acceptance 6 are built here; the section is done when all five build with no further change to the library. `docs/loader.md` is completed here.

| Description | Ceiling |
|---|---|
| Sensors joined by a kernel of road distance above a threshold; a reading every five minutes; predict the next twelve | 10 |
| Interactions of users with items over time; predict the next item among sampled negatives | 7 |
| Accounts and transfers as parallel edges with amounts; the label of a transfer arrives thirty days later and is then known to later tasks | 8 |
| Papers citing papers; a paper sees only older papers; predict the venue | 6 |
| Triples of head, relation and tail with thousands of relations, some about other triples; predict the tail | 5 |
| Many small graphs in one file with a column naming the graph, nodes identified by row number; a target per graph | 6 |
| Events that each touch several objects of several kinds, each with a role | 8 |
| Visits in order per patient, each with sets of codes; codes under a table of parents, all ancestors joined | 12 |
| Trees of nested records with ordered children, and two more sets of edges over the same nodes | 9 |
| Papers joining all their authors as one edge; predict the next set | 6 |
| A game where several players act at once and each sees part of the state | 12 |
| Clicks cut into sessions by a gap of thirty minutes, with edges between consecutive clicks | 6 |
| Trips as ordered sequences of road segments; predict the duration of a trip | 8 |
| Machines with logs of sensors; predict the days until failure, where most had not failed when recording ended | 8 |
| Drugs that are each a graph of atoms; predict which pairs interact | 8 |
| Quarterly figures that are revised later; predict with what was believed on the day | 6 |
| Impressions in which a policy showed items in three positions, each with the probability of being shown; the actor is the policy, the outcome is the click | 7 |
| Decisions where only the chosen set of actions was logged, each action with targets in the graph and a number as parameter; the other options are drawn by a function | 9 |
| A database with declared keys and views: a target, the columns of time, and who is eligible | 6 |

**6. Scale and resources**

The machine: 64 GB of memory, 24 threads, one GPU with 32 GB, 820 GB free on the disk of the runtime home on 28 September.

6.1 Budgets. A build, a derivation and a fit each take stated budgets of memory, of GPU memory and of disk from the statement of 1.6, and stay within them. `graphspec/resources.py` reads them from the statement, not from constants. The variant cache has a stated limit and a registered rule of eviction; shipped: least recently used. Check, without a fit: a synthetic build three times larger than its budget of memory completes with its measured peak under the budget; a cache filled past its limit evicts by the rule.

6.2 Storage that is not loaded whole: columnar files, compressed, mapped into memory, read by range of time and by node. Facts store the time from which they were known only where it differs from the start of their interval. Check, without a fit: reading one node's neighbourhood from a synthetic graph larger than the budget of memory stays under the budget.

6.3 Partitioning. A graph is cut into parts by a registered rule; shipped: by range of time, and by hash of key. A batch reads only the parts its tasks touch. Check, without a fit: on a synthetic graph of twenty parts a batch of tasks of one range of time reads the parts of that range and no other.

6.4 Sampled batches from one large graph: neighbourhoods around the tasks of a batch, with nothing later than each task's time, a stated number per hop and a registered rule (shipped: most recent, uniform). Batches in order of time for streams of events. Nodes of very high degree are capped by a stated rule. For decisions, the options of a decision are scored on the sampled neighbourhood of their targets. Check: on a synthetic graph larger than the stated budget of memory, every sampled edge is no later than its task; a fit on sampled batches of that graph, of at most 200 nodes with the budget stated below its size, reaches a loss within 5% of a fit on the whole graph. Budget of the check: 2 seconds.

6.5 The path to the GPU: features fetched by batch, pinned memory, stated precision, stated number of sampling workers. Dense algebra moves to the GPU only where the statement says so. Check in the suite, without a fit: with dense algebra set to GPU and none present, an error names the setting; a batch fetched to a stated device has the stated precision. The fit on a graph larger than its budget of GPU memory is checked in 10.1.

6.6 Benchmarks. A verb `benchmark` records, per operation and per dataset, rows per second, peak memory, GPU memory, mean GPU use and disk written. Conversion of a graph to the objects of PyG is built here, so that the same data can be timed through the loaders of PyG on this machine as the outside reference. Requirements fixed now, checked in 11.1: every dataset marked `limit` builds within 48 GB of memory, within 24 hours, and within disk of three times its packed size; a fit on the largest graph runs with sampled batches within 28 GB of GPU memory; rows per second of loading and of sampling are at least those of the reference on the same dataset. Check in this section, without a fit: the verb runs on a synthetic graph and writes every measure; a synthetic graph converted has the same nodes, edges and values.

6.7 Checks of speed inside the suite: each registered operation on a synthetic graph of 10,000 nodes and 50,000 edges within 0.2 seconds.

**7. Datasets return as scripts**

7.1 The first dataset that was an adapter, as a script in `local/`, written from the adapter code in the copy of 1.1 and from its source file. Entities with a period of validity are nodes with intervals, and the edges to their holders carry the same intervals. Intended differences from the staged tables kept in 1.2: intervals exist; four features of the holder that count its edges are gone, being measures of 2.9; the flag of why a period ended stays where it was known before the time of the task; members of splits.

7.2 The second, as a script in `local/`, written the same way. Intended differences: the five values and the fact listed in the table of faults as outcomes are gone; missing values follow stated rules; merged tables are prefixed; events are events; references that do not resolve are counted; members of splits. No column of the source that records whether an action validated or succeeded is read.

7.3 The third, as a script in `local/`, written the same way. Intended differences: parallel edges are kept; regenerated and recorded options carry no feature that tells them apart; decisions and options are declared by 3.1, each option with where it comes from; members of splits. No column that records whether an action validated or succeeded is read, and runs are not chosen by any such column.

7.4 One script per public dataset named in the column Use of the table of public data, in `local/`, under acceptance 5. The databases with declared keys are converted by 5.8, each within six statements; the collection of about 80 databases is converted by one script that differs per database only in target and eligibility. Waits on B2. Done when each builds and validates, started detached.

7.5 Check of 7.1 to 7.3: the staged tables kept in 1.2 are read by a converter kept in `local/`, run detached; the library's difference between the old and the new graph lists only the intended differences; each script built twice gives equal fingerprints; the report of acceptance 2 is given, and the count of statements of each script is written into this plan as its ceiling.

7.6 `README.md` is rewritten: one path, from a source to a search. Check: every command in it is run by a test.

7.7 The test of names of acceptance 3 covers its full list. Check: it passes.

**8. Tuning**

8.1 The record of a search. `plan.json` becomes version 10 and holds the statement of 1.6 as it stands at this point, with, for a search, selector, mix, number of random trials, random parameters, seconds per trial, bounds and limit of trials. Check: the `plan.json` of a synthetic search holds every field of the statement, and a search whose `plan.json` lacks one stops with an error naming it.

8.2 Public surface. `graphspec` exports the builder, `Stage`, the schema, the operations and the registries, `validate`, `inspect`, `describe`, `derive` and `compare` (the difference of 2.9). `tuning` exports `build_bank`, `open_bank`, `statement`, `fit`, `search`, `load` and `predict`. `GraphTables`, `write_task` and `configuration` are removed. The command has the verbs `inspect`, `build`, `check`, `search`, `dashboard`, `synthetic`, `benchmark`, `describe`; `panel` and `track` are added in 8.3; `README.md` is corrected here; the verb that exported staged tables to a graph database is removed with the helper it used. Importing the library reads no environment variable, creates nothing and changes nothing; the runtime home is set by the command and by an explicit call. `load` opens a finished trial without a training session; `predict` takes a graph, tasks that need no stored target, and a stated device. Check: a synthetic script that imports `graphspec.infra` fails a test; importing in a process without `GRAPHSPEC_HOME` succeeds and leaves the environment unchanged; on the scoring tasks of a synthetic trial `predict` equals the stored predictions within 1e-4 on CPU and on GPU.

8.3 Measures: registry, utility and combination. Measures of how well a model tells options apart are registered. Each states what it needs: options; outcomes on options that were not chosen; probabilities; a model of the state alone; several fits. A measure whose needs are not met stops with an error. Measures that order options read the utility of the kind of task and nothing else. Every measure is recorded with the number of options it was computed over and with an interval from resampling decisions. The statement says which measures are computed, which select, and how they combine; ways of combining are registered; shipped: one alone; a weighted sum; thresholds that a trial must meet; several objectives kept apart, from which the search returns the trials that no other trial beats on all of them. The measures of sequences written on 28 September are rebuilt from the copy of 1.1 as registered measures for path targets; their constants in days, their number of series and their wording come from the task's declaration. The verbs `panel` and `track` are rebuilt here from the copy of 2.1. Check: a measure registered by a test is computed and selects; each way of combining orders three synthetic trials as stated in the test.

8.4 Measures that need only options and the outcome of the chosen one. Definitions, D being the decisions of the scoring split, u the utility, and loss the loss of the fit in scaled units:

| Measure | Definition | Rewards |
|---|---|---|
| Substitution gap | Mean over D of the mean over the available options not chosen of loss with that option minus loss with the chosen one; availability that records whether an action validated is not read; options of content equal to the chosen one are left out and counted | being right |
| Permutation gap | The same, with the option of another decision of the same graph put in place of the chosen one, only where every target resolves and is available; the share of decisions with no exchange possible is recorded | being right |
| Ablation gap | Loss of a model of the state alone minus loss of the model; needs a second fit | being right |
| Rank correlation of preference and residual | Spearman over D of: u of the chosen option minus the mean u over the decision's options; against realised u minus the u predicted by a model of the state alone. Recorded with the name of the generator where options are regenerated | being right |
| Percentile and reciprocal rank of the chosen option | Where the model places the chosen option by u; ties take the mean rank | agreement with the actor |
| Spread | Highest minus lowest prediction over the options, averaged over the predicted values of the decision, then over D | movement |
| Deviation over options | Standard deviation of the predictions over the options | movement |
| Mean difference between pairs | Mean absolute difference over pairs of options | movement |
| Separation against noise | Share of pairs of options whose absolute mean difference over S fits exceeds k times its deviation over those fits; S and k from the statement | movement beyond noise |

Check: on synthetic decisions whose outcome depends on the option by a known rule, a model given the rule scores above a model that ignores options on every measure that rewards being right; a fixture of predictions that move at random scores above the first on the measures of movement and below it on the measures that reward being right; a fit whose options are identical records a spread of 0, and the score of the search of 28 September applied to it uses the floor of 1e-6.

8.5 Measures that need outcomes on options that were not chosen: regret of the option the model ranks first; rank correlation of predicted and realised u within a decision; share of pairs ordered correctly; normalised discounted gain at k. Check: on the synthetic decisions of 8.4 with every outcome known, the model given the rule scores a regret of 0.

8.6 Measures that need probabilities of the logged choice, recorded or estimated by a registered model of the logging policy fitted on the splits trained on: value of choosing by the model, weighted by inverse probability, self-normalised, clipped at a stated value, and doubly robust, each a separate measure, each recorded with the effective number of decisions and the share matched; the estimator for slates by pseudo-inverse; fitted evaluation of values, which reads the links between decisions of 3.1; the curve of uplift and its area; difference in outcome between the nearest pairs of states with different chosen options. Check: on synthetic logs drawn under a known policy, each estimator is within its interval of the true value of the policy evaluated.

8.7 Families are registered with what they support: kinds of task and of target, decisions, entities and graphs not seen in training, sampled batches, CPU and GPU. The search offers a family only where it is supported. A bank may have no scoring split when the statement says the fit only fits. Check: a family registered by a test is offered for the kinds it states and for no other; a fit on a synthetic bank with no scoring split completes.

8.8 Mechanisms that make a model use options. Registered, and stated by the fit or searched. Shipped: a separate route for the option into the head; a prediction formed as a product of an encoding of the state and an encoding of the option; predictions of one decision centred on their mean; a model of the state alone fitted first, with the rest fitted to what it leaves; removal of part of the history in training; an added task of recovering the chosen option from state and outcome; a penalty on options far from logged ones; a penalty by the disagreement of several fits. Whether options without an outcome are read as inputs by a loss is a field of the statement: never; only to centre; or also for contrast, where the label is which option was chosen and no outcome of another option is a target. Check: on synthetic decisions where the state explains 95% of the variance and the model without the mechanism has a substitution gap whose interval holds zero, each of the first six mechanisms on its own gives a gap whose interval lies above zero; each of the two penalties lowers the quantity it penalises. Budget of the check: 4 seconds.

8.9 Regenerated options are checked for what they give away. A classifier is fitted to tell chosen options from regenerated ones given the state; the area under its curve on the scoring split is recorded with every fit that uses regenerated options. The statement says the area above which the fit stops. Check: on synthetic options with a feature that marks the regenerated ones, the fit stops with an error naming the feature of greatest weight.

8.10 Seeds. The draw of the selector is seeded by the seed of the statement and the trial's index. The candidate set is made by a fixed number of seeded streams whatever the number of cores. One record per search states how each selector treats failed trials; the treatment itself is unchanged. Check: two searches of one statement on one machine propose the same first twenty trials.

8.11 Every derived graph and every trial records its statement, the versions of its operations, its seeds and the fingerprint of its source. Check: a test reads them back from a synthetic trial.

8.12 The statement is complete. It holds every choice of a fit and of a search. Check: a test collects every field added by 1.6 and by the subsections after it and finds each in the record and in `plan.json`; the list of fields is written into this plan, and a test fails if the record holds a field not in that list.

**9. Descriptions and tracking of models**

What a model reads, what it responds to, and how well it predicts are written by the library for every fit. Nothing here is a score; what selects a model is stated in the statement (8.3).

9.1 What a family reads. Every family declares its inputs: which references of a task it starts from; how many hops it reads; whether it reads features of nodes, kinds, keys, values of edges, the relation of an edge, the features of the task, its category, other inputs of the task, and time; how it orders what it reads and how many items it keeps. The declaration is held true by a test run for every registered family: changing an input the family declares unread leaves every prediction equal; changing each input it declares read changes a prediction for at least one task. A family without a declaration is not registered. Check: the test passes for every shipped family, and fails for a family registered by the test with a false declaration.

9.2 What a fit can read. For a fit, the reach of 4.3 and the declaration of 9.1 are joined: for each field of an option's content and of the state, whether it can arrive at the prediction, by which route, or where it stops (not connected, beyond the hops read, of a sort the family does not read, cut by the number of items kept). Written before training starts. Check: on synthetic decisions the answer equals the known one for a family that reads one hop and one that reads three.

9.3 What a fit responds to. After training, on the scoring split, using the forks of 2.7: for each field of an option's content, the options of a decision are made alike in everything but that field, and the range of the predictions is recorded; then the share of the whole range over options that each field accounts for, the share of pairs of options of different content with equal predictions, and the fields whose influence is zero. For regenerated options, how often the recorded option is the highest or the lowest, beside the share expected by chance. How the range is spread over decisions: the share carried by the largest tenth of them. Check: a synthetic model built to respond to one field alone shows that field at a share of 1 and the others at 0, within the interval of the measure; budget of the check 2 seconds.

9.4 How well a fit predicts. The loss on the scoring split beside the loss of each reference registered for the kind of target; shipped for numbers and paths: no change, and the mean of the splits trained on. For path targets, per series and per band of offset, with the number of points trained on and scored on in each band, the correlation of prediction with truth, and the deviation of the predictions beside that of the truth. Check: on a synthetic path target with a known signal, the numbers equal those written out in the test.

9.5 The card of a fit. One document per fit, written when it ends, holding in this order: the statement; the description of the variant (4.1, 4.2); what the fit can read (9.2); what it responds to (9.3); how well it predicts (9.4); the measures of 8.3 with their intervals; failures and their cause in words. The dashboard shows it; `describe TRIAL` prints it. Check: for a synthetic fit a test reads every part of the card and compares it with the known truth.

9.6 Tracking over a search. One table with a row per trial and the answers of the cards as columns, which the dashboard filters and sorts and `describe SEARCH` prints. Rules of eligibility for choosing a model are stated in the statement as conditions over these columns, such as that a named field of an option is read and has influence above a stated share. The conditions are a field of the statement, added to the list of 8.12. Check: on a synthetic search of twenty trials, a condition stated over two columns leaves the trials the test names.

9.7 The list of questions. Every question asked about a graph, a model or a search is kept in this plan with the verb or call that answers it. A question that needed a reading of code, or a script written for the occasion, is a defect, and is removed by adding to a description. The list starts with the questions of 28 September:

| Question | Answered by |
|---|---|
| What does this graph hold | 4.1 |
| What did each operation remove | 4.2 |
| Which operation and setting removed these edges | 4.2, 4.3 |
| Are the options connected to the task at the time of choosing | 4.3 |
| In which trials of a search are they | 4.4, 9.6 |
| What can this model read of an option | 9.2 |
| What does this model respond to, and how much | 9.3 |
| Why is its differentiation high | 9.3 |
| Is it telling the recorded option from regenerated ones | 9.3, 8.9 |
| Does it predict better than predicting no change, and at which offsets | 9.4 |
| Which trials are eligible under a stated rule | 9.6 |

Check: on a synthetic search, a test asks each question through the verb or call named and compares the answer with the known truth; acceptance 7.

**10. A search that runs without an operator**

Carried from the old plan with the changes named in the table of differences.

10.1 The command refuses to start `build`, `check`, `search` and `benchmark` when the library resolves inside a git checkout. `scripts/graphspec.ps1` no longer uses the checkout as working directory. The library is reinstalled before a search. Calls from Python, which tests use, are not refused. `README.md` is corrected here. Check: a test starts the command from a checkout and reads the refusal.

10.2 Resume takes only `--output`. Everything else comes from the statement in `plan.json`. A flag given replaces the stored value and emits the existing `*_replaced` event. The reset of workers, variant budget and limit on resume (`plan.update(workers=..., variant_bytes=..., limit=...)`, `tuning/run.py:108` on the working tree) is deleted. Check: a test resumes with `--output` alone and reads the stored values back.

10.3 `RUN/settings.json` holds the fields of the statement that may change between trials: selector, mix, number of random trials, random parameters, seconds per trial, bounds, limit of trials, measures that select and how they combine. Flags of start and resume write it. The runner reads it at every boundary between trials. The first resume under the new code creates it from `plan.json`. STOP stays the one way to stop. Check: a test changes a setting between two trials and reads the replacement event.

10.4 `search run` and `benchmark` start their work as a process that outlives its launcher, with output to a log in the run, and `--foreground` for tests. `scripts/detached.ps1` is deleted. Check: a test starts a search, ends the launcher, and finds the runner alive.

10.5 A test that an interrupted trial is retried as a new attempt.

10.6 The dashboard writes STOP and `settings.json`. Check: a test posts both and reads the files.

**11. Runs on real data**

Started detached. Waits on B2.

11.1 `benchmark` on every dataset marked `limit`, and the same datasets through the loaders of PyG; equality of derived arrays for 1 and 8 workers on the largest; a fit on a graph larger than its budget of GPU memory completes within it. Every check the suite skipped for lack of a GPU or a database server is run and passes. Done when the measures are written into this plan and the requirements of 6.6 are met.

11.2 One fit of 1,000 steps on every dataset named in the column Use. Done when each completes within its budgets and its measures are written into this plan.

11.3 One search of 20 trials on each of the scripts of 7.1 to 7.3, and one on a public dataset with recorded options and one with regenerated options, each stating measures of 8.4 that reward being right among those that select. Done when their results are written into this plan.

**12. Cleanup**

12.1 Delete the writes of `runner_timing.json`, `maintenance_timing.json` and `timing_workers/` in `tuning/run.py` and `tuning/training.py`. Check: a synthetic search leaves none of them.

12.2 Fallbacks. First, every site under `graphspec/` and `tuning/` where code puts a value in place of one that is missing is listed in this plan by file and line; done when the list is in the plan. Then each is removed or becomes a stated rule; done when a test that searches for the listed patterns finds none. A rule that becomes a field of the statement is added to the list of 8.12.

---

**Public data**

Stored under `D:/graphspec-runtime/sources`, outside the repository and not used by the library until section 7. `manifest.json`, `manifest-relational.json` and `manifest-open.json` give origin and licence; the `results` file of the same name gives size, checksum and time of download of every file, or the error of a file that failed; `fetch.py` resumes and skips what is complete. `test`: a small part is cut for the test suite. `limit`: used whole by section 11.

State on 2026-09-28 at 18:47: 14 of 26 sets complete, two partial, two queued, one refused at listing, seven not yet fetched.

| Set | Folder | What it breaks or stresses | Use | Size | Licence as its source states it | On disk |
|---|---|---|---|---|---|---|
| A company's database of sales, purchasing, production and staff: 68 tables, 173 foreign key clauses, 90 views | `relational-adventureworks/` | Conversion from declared keys and views; dated orders; eligibility by customer | test: `AdventureWorks` | 17 MB | MIT for the original samples; the Postgres port states none | complete |
| A film rental database: 37 foreign keys, 8 views | `relational-pagila/` | The same, small enough for the suite | test: `pagila` | 13 MB | PostgreSQL licence | complete |
| An open sample of a hospital database, 100 patients: 52 foreign keys | `relational-mimic-demo/` | Visits in order with sets of codes; declared keys | test: `mimic-iv-demo` | 16 MB | Open Database License | complete |
| A music catalogue, full export: 374 tables, 762 foreign keys, 10 views | `relational-musicbrainz/` | Conversion from declared keys at a size where work per table fails | limit: `musicbrainz` | 8.3 GB | Core data public domain; derived, editor and statistics parts not for commercial use | complete |
| A collection of about 80 unrelated databases with declared keys | `relational-bird/` | The same conversion on many schemas, differing only in target and eligibility | limit: `bird-train`, `bird-dev` | 9.3 GB | CC BY-SA 4.0 | complete |
| Temporal graph benchmark, 17 datasets | `tgb/` | Streams of events with times in seconds; targets that are proportion vectors; links among sampled negatives; statements about statements; up to 53M edges | test: `tgbn-trade`; limit: `tkgl-wikidata`, `thgl-myket`, `tgbl-flight-v2` | 24.46 GB | MIT for the repository; terms per dataset not stated | complete |
| Relational benchmark, first release, 7 databases | `relbench-v1/` | Tables with keys and times; tasks as separate tables; times of prediction set by the task; several tasks per database | test: `rel-f1`; limit: `rel-amazon` | 10.29 GB | CC BY-SA 4.0 on the card; the sources beneath carry their own terms | complete |
| Relational benchmark, second release, 9 databases | `relbench-v2-extra/` | The same, with tasks that fill in a cell of a table | limit: `rel-ratebeer` | 3.48 GB | same | complete |
| Molecule sets, 2 | `tudataset/` | Many small graphs; nodes identified by row number; targets per graph; no time | test: `MUTAG`; limit: `ZINC_full` | 39 MB | not stated | complete |
| Signed trust networks, 3 | `signed/` | Targets on edges; signed weights | test: `soc-sign-bitcoinotc`; limit: `soc-sign-epinions` | 3.5 MB | not stated; citation requested | complete |
| Layout of traffic sensors, 2 networks | `traffic-sensors/` | Edges from road distance | test: `distances_la_2012` | 6.6 MB | MIT for the repository | complete |
| Open graph benchmark, 10 datasets | `ogb/` | One large graph, the largest of 111M nodes; classes; links; a knowledge graph; graphs of code with a sequence as target | limit: `papers100M-bin`, `products`, `wikikg-v2`, `code2` | 68.1 GB | MIT for the repository; terms per dataset on its site | 9 of 10 files |
| Readings of traffic sensors, 2 networks | `traffic/` | A signal at regular times on a graph that does not change; several horizons | test: `METR-LA`; limit: `PEMS-BAY` | 160 MB | CC BY 4.0 | not yet |
| Object-centric event logs, 7 logs | `ocel2/` | Events touching several objects of several kinds; attributes that change | test: `5_recruiting`; limit: `3_o2c` | 21 MB | CC BY 4.0 | listing refused; to be fetched again |
| Logged impressions, sample | `open-bandit-sample/` | Options recorded as shown, with the probability of each and three positions | test: `random/all` | 25 MB | Apache-2.0 for the repository | not yet |
| Logged impressions, whole | `open-bandit/` | The same at 26M rows | limit: `open_bandit_dataset` | 413 MB | not stated | not yet |
| Randomised trial of a treatment | `uplift/` | The action is an input and was assigned at random; no graph given | limit: `criteo-uplift-v2.1` | unknown until fetched | CC BY-NC-SA 4.0, not for commercial use | not yet |
| Industrial graphs with tabular features, 14 | `graphland/` | Columns of mixed types with missing values; numbers and classes; up to 33M nodes | test: `tolokers-2`; limit: `web-traffic` | 3.62 GB | Apache-2.0 | not yet |
| Replays of a strategy game | `starcraft/` | Decisions of two players with many units acting at once; only the chosen actions are logged, so options are regenerated | limit: `2024_HomeStory_Cup_XXVI` | 12.5 GB | CC BY 4.0, under the publisher's licence for machine learning | not yet |
| Targets on sets of nodes inside a base graph, 4 datasets | `subgraphs/` | A task whose reference is a set of nodes | test: `PPI-BP`; limit: `EM-USER` | 15.9 GB | not stated on the download; MIT for the authors' repository | complete |
| Transfers between accounts, each labelled | `transactions-labelled/` | Targets on edges; parallel timed edges with amounts | test and limit: `HI-Small_Trans` | 91 MB | CDLA-Sharing-1.0, stated by the publisher; this copy is a mirror | complete |
| Transactions and wallets of a blockchain | `transactions-wallets/` | Two kinds of node with labels on both; edges between both kinds | limit: `ellipticpp` | 2.2 GB | unknown | complete |
| 8,600 traffic sensors over five years | `traffic-large/` | A signal on a fixed graph at a size that does not fit in memory | limit: `LargeST` | 37 GB | CC BY-NC 4.0, not for commercial use, stated by the publisher; this copy is a mirror | partial, being fetched |
| Open sample of a database of intensive care stays | `hospital-icu-demo/` | Stays in order with measurements at irregular times | test: `eicu-demo` | 137 MB | Open Database License | queued |
| Generated records of 575,000 patients | `hospital-synthetic/` | Visits with sets of codes at scale; a table of 97 GB | limit: `synthea-575k` | 143 GB | MIT | queued |
| Temporal hypergraphs, 7 | `hypergraphs/` | Edges joining sets of nodes, with times | test: `email-Enron`; limit: `coauth-DBLP` | unknown until fetched | not stated | not yet |

No dataset with decisions is on disk yet: `open-bandit-sample`, `open-bandit`, `uplift` and `starcraft` are among those not fetched. Section 3 and 8.3 to 8.9 are checked on synthetic decisions; 11.3 waits on B2.

Where a licence is given as not stated, none was found on the page of the download; that is not permission. Databases arrive as dumps and scripts and are loaded into a database engine in 7.4.

Two datasets are not taken, and the table above holds what stands in for them. The full hospital database is given only to credentialed researchers; the two open samples and the generated records stand in for it. One dataset of targets on sets of nodes is given by its publisher only on request; `subgraphs/` stands in for it.

Left out by choice: MAG240M (167 GB), from the same source as the open graph benchmark.

**Order and current state**

Order is 1 to 12. Every section waits on B1, which holds no longer: no search is running.

As of 2026-09-28 at 21:00: `main` is at `32af441`. The work of 28 September is uncommitted and stays so: 1.1 keeps a copy of it; 2.7 replaces `tuning/offers.py`; section 3 and 8.3 to 8.9 replace the scoring of spread and the modules of measures; 7.3 replaces the changes to the third adapter. The search on the third dataset was killed by the owner at 101 recorded trials, which remain under the runtime home.

As of 2026-09-28 at 22:50: on the owner's instruction the whole working tree, untracked files included, was committed as `9794d2c` on the branch `snapshot-20260928`; `main` is unchanged. This replaces the statement above that the work stays uncommitted. The work of this plan is done on the branch `generalise`, made from that commit.

| Subsection | State | Evidence |
|---|---|---|
| B1 | clear | No search running, checked 28 September at 22:40 |
| B2 | clear | Audit of 28 September at 22:53 |
| 1.1 | done | Copy at `D:/graphspec-runtime/ground/worktree-20260928`; fingerprint of source and copy `391a6b3d3531b6d6930e357df5bc8dd1816367d56156f64ab888f1ab5a76b319` over 1,921 files, `__pycache__` and `*.pyc` left out; recorded in `D:/graphspec-runtime/ground/README.txt` |
| 1.2 | done | Staged tables under `D:/graphspec-runtime/ground/staged`, fingerprints of the staging folders: first adapter `e4d967eec17a6d8bb04d9770655622802f95cf3c321679b20aee59e699e890cc`, second (4,000 graphs) `6613eba39520ead0a45d130eccdd543302a166f080a15c83a53e95e5dfe9c0c0`, third (runs 1 to 57) `d1f18fa11a87283cb37073962a7c453b1f7c438f23a7b841be7cfe6528fec400`. Fixture under `D:/graphspec-runtime/ground/fixture`, fingerprint `5551d5249accdd2f35a4666435b4242d7ded081493c1577b0c0bd51b2bd3a0c2`, equal over two runs; its ten configurations are in `fixture.json` there |
| 1.3 | done | `566cc81`; four files differ from `main`; 294 passed in 10.0 s |
| 1.4 | done | `258151d`; 272 passed in 10.9 s |
| 1.5 | done | `8c46118`; 276 passed |
| 1.6 | done | `4ba4f68`; 294 passed |
| 1.7 | done | `b805c21`; 297 passed in 10.6 s |
| Review of section 1 | done | Repairs in `26acf81`, `451f229`, `31a29a5`; 392 passed in 11.3 s |
| 2.1 | done | `6072ce9`; 336 passed in 12.2 s |
| 2.2 | done | `42b90b1`; 431 passed in 13.0 s |
| 2.3 | done | `9249781`, and `2a23b17` for the time of the suite; 470 passed in 11.4 s |
| Hand-over | written | `D:/graphspec-runtime/ground/HANDOFF.md` holds how the work is run and what is open; updated on 29 September at 03:30 when the agents running on Fable stopped at its usage limit and were replaced |
| 2.4 | done | `b616dd1`; 478 passed in 12.7 s; ruff clean |
| Review of 2.1 to 2.3 | done | Two reviewers; reports, findings and reproductions in `D:/graphspec-runtime/ground/review-2.1-2.3` |
| Repairs of the review, suite and robustness | done | `73780d2`, `ed7897b`, `4dc9353`, `05b172f`, `28ec290`, `46d894e`, `fe65a1a`; 497 passed in 13.9 s, serial run 497 passed in 28.4 s; ruff clean; 1.70 GB of suite caches deleted by exact path. Cause of the stale caches: numba names its helpers with a counter restarting at 1 in each process, so a kernel loaded from cache was linked to a different helper compiled fresh; the counter now starts at a random number per process |
| Follow-up of those repairs | to do | Where a step covers several events about one entity the view shows the latest without a stated rule; `snapshot` with a context of earlier periods is offered and refused at fit; the build's two process pools merged into one to recover suite time; the redundant `source` in the variant key removed |
| Repairs of the review, leakage | to do | Encoders fitted on rows the trained splits do not see; matured targets cross observers and splits; `without_targets` hides every edge named by a reference, not the edges that are targets |
| Repairs of the review, references | to do | Seven families read two references by position and drop a third without a word; the kind of a task reaches no model |
| Repairs of the review, relations from a column | to do | One search parameter, one stored array and one set of weights per value of the column |

**Deviations**

One line per place where the plan was wrong or silent: the subsection, what the plan said, what was done, why.

| Subsection | The plan said | Done | Why |
|---|---|---|---|
| History, Order | The work of 28 September is not committed | Committed whole as `9794d2c` on `snapshot-20260928`; `main` unchanged; work proceeds on `generalise` | Owner's instruction of 28 September, to be able to revert |
| 12 | A fragment of a removed section on a hand-over followed 12.2 | Removed | The hand-over was cancelled; the fragment was left by a faulty deletion |
| 1.1 | The copy's fingerprint equals that of the working tree | Fingerprints leave out `__pycache__` and `*.pyc` | Bytecode is not source and changes when code runs |
| 1.2 | Ten configurations listed in this plan | Listed in `D:/graphspec-runtime/ground/fixture/fixture.json`; the joined route is stored for two of them as well | 2.7 compares the engine against the joined route |
| 1.3 | `git status` shows only two files and their tests | Checked as the difference from `main`, which lists those four files | The tree was committed, so `git status` is empty |
| 1.3 | Two corrections applied as they were | The single-thread helper was given a timing log | Every function logs its time |
| 1.4 | Not stated | The test suite takes a lock of its own, `heavy-tests`, in place of the machine-wide lock | A detached build held the lock and the suite could not run beside it |
| 8.3, 5.12, 2.9 | Refer to the copy of 2.1, the cases of 4.1, the forks of 3.7 | Read as 1.1, 3.1 and 2.7 | Left from a renumbering |
| 1.2 | Fingerprints of the staged tables | The fingerprint is of the staging folder, not of the whole output | The build record beside it holds the seconds the build took |
| 1.2 | Not stated | Arrays of the fixture are written with a fixed date in the archive | The usual writer stamps the time and the fixture would not reproduce |
| 1.5 | A synthetic fit completes on CPU | The fit in the suite uses the suite's running packer and does not start its own | Starting one costs 1.8 seconds on this system |
| 1.5, 1.6 | The device is an argument, then a field of the statement | For dense algebra it travels as a required field of the limits passed to operations and to workers | One route; nothing is read from the environment |
| 1.6 | Every field is required | A bank is opened without a device and stops when asked to derive without one | Most uses of a bank derive nothing |
| 1.6 | The statement is one record | `plan.json` keeps version 9; each trial writes its statement beside its result | 8.1 changes `plan.json` |
| 1.7 | A module that defines a function | Functions compiled to machine code are not counted; their callers are timed | Compiled functions cannot log |
| 2.1 | Each kind of node has its own width | All columns are encoded into one table with a block per column; the first layer of a family holds a block of weights per column | Every family sums vectors across kinds before projecting; none had to be rewritten. Revisit in 6.2 if the width limits scale |
| 2.1 | Equality with the fixture after sorting by key | Equality of what rows resolve to, with minus zero equal to zero | The old form wrote minus zero for an absent negative value |
| 2.1 | Splits trained on | `build` takes them as required and accepts only `train` | 2.3 makes a split any name |
| 8.1, 10.2 | A flag given on resume replaces the stored value | Until 8.1 and 10.2 are built, a resume whose seed, seconds, workers, cache size or batch size differ from the stored ones stops naming the field | One source for each value; 10.2 restores replacement with its event |
| 8.1 | Bounds of a search | Until 8.1, bounds that hold the seconds of a trial are refused, so a search does not vary the seconds of a trial | The statement holds the seconds; 8.1 puts bounds in the statement and restores the range |
| 2.6 | Key of the variant cache | The key holds the device as stated since the repairs | Dense algebra differs between CPU and GPU from 1,024 rows |
| Rules | The suite runs every test | Tests marked `gpu` are left out of the default run and run in section 11 | The suite stays off the GPU |
| 2.2 | Graphs of one bank may have different units | The unit is a name per graph; nothing converts between units | Nothing compares times across graphs |
| 2.2 | Not stated | Times are whole numbers in the unit of the graph | A finer unit is stated in place of a fraction |
| 2.2 | Several facts about one entity at one time are allowed | All are stored and read back; the view shows one, chosen by a stated rule | The engine reads one per entity per step until 2.7 |
| 2.2 | A fact known later is invisible to a task before | A step of a view shows the facts as the task of that step saw them; a later task does not see in earlier steps what became known after them | A history per task needs the forks of 2.7 |
| 2.2 | A bank of two graphs with different units | The case is a bank of five graphs that alternate between two units | One bank is shared by the cases of the check |
| 2.2 | Not stated | A task without an observer sees every fact | No observer stated means no restriction stated |
| 2.3 | Names and number of references are stated per kind of task | References are named once in the schema and each kind picks from them | One name means one thing across kinds |
| 2.3 | Models receive references as a set keyed by name | One family reads them by name; the others read the first three node references in stated order and stop on kinds they cannot read | To be completed with 8.7 and 9.1, where families state what they read |
| 2.3 | A kind of task has the columns listed | A kind also states what an absent reference does, whether it is an event between two references, and what a known target becomes | The check's cases need each stated |
| 2.3 | A bank with targets on nodes and on graphs together | Targets that are numbers and targets that are paths are not mixed in one bank | Kinds of target are 2.10 |
| 2.3 | An optional weight | A kind that states no weight has weight 1 | Absence of a weight means equal weight |
| 2.4 | An edge may be the endpoint of another edge | An endpoint may name a kind of set edge; an edge that must be an endpoint is stated as a set edge of two members with roles | An edge needs a key to be referred to |
| 2.4 | Not stated | No sort of reference was added; a reference of a node sort may name kinds of set edge | A set edge is keyed like a node |
| 2.4 | Not stated | The members of a set edge and the graph a node holds are stated once and carry no time | The simplest statement that serves the check |
| 2.4 | A node's graph is encoded by a stated encoder | Shipped encoders `pooled` and `propagated` are fixed and computed in the bank, not learned | The check asks for a stated encoder, not a learned one |
| 2.4 | Not stated | An edge about a set edge states a registered rule for times when that edge does not hold; `stop` and `lapse` shipped | No silent behaviour |
| 2.4 | Not stated | Relations made by the forms of storage are named `<kind>.<role>` and `<kind>` | Relation names become file names, and `:` is not allowed in them on Windows |
| 2.4 | Not stated | The database source refuses schemas with set edges or held graphs; they are staged | The database reader is rebuilt in 5.3 and 5.8 |
| 2.4 | Not stated | Schema version 4 | The shape of the schema gained required fields |
| Repairs | Not stated | Numba's counter of compiled functions starts at a random number in each process, set in graphspec's package init | Cached and fresh compiles in one process shared names and linked the wrong code |
| Repairs | Not stated | Suite homes are `workers/<checkout>-<hash>/<worker>`; the suite deletes what it made there after a run except `cache/numba` | Without the compiled cache every run recompiles for about 54 s |
| Repairs | Not stated | Caches of views and variants are keyed by the manifest and the digest of the library's sources | A cache never serves what other code made |
| 2.2 | A step shows the facts holding at its time | A step covers the times after the previous step up to its own; an event is shown at the first step at or after its time | Events between task times were never shown |
| 2.2 | Facts with the same time are ordered by a stated rule | A rule for ties is required only between facts about one node or one edge | Order matters only within one entity |
| 2.2 | Not stated | `time/several` has no default and exists where the bank holds overlapping facts about one entity | Registering a rule silently changed every configuration |
| 6.5 | Precision of model inputs is stated | Until 6.5 inputs are stored at 16 bits; a value past 65504 stops where it is encoded, naming graph, kind, column, value and encoder | The contract states the limits |
