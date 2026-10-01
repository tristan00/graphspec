**Graph spec core**

| Where | What is there |
|---|---|
| `schema/` | What a graph is stated to hold: the schema, types and encoders of columns (`schema.py`), time and its rules (`time.py`), references (`references.py`), a bank's declaration of relations (`declaration.py`), what an operation adds to it (`added.py`) |
| `facts/` | The truth graph as a bank stores it (`stored.py`), and its check (`check.py`) |
| `views/` | What a task sees at a step: the one guard of visibility and the rules of what a split sees (`visible.py`), steps and the view (`view.py`), the view as spans and its checks (`graph.py`), spans (`spans.py`) |
| `build/` | Stating a graph (`GraphTables` in `calls.py`, `forms.py`), writing it (`Stage` in `output.py`), reading it back (`staged.py`) and parsing it into facts (`parse.py`) |
| `sources/` | Reading sources: LadybugDB (`ladybug.py`), the synthetic source (`synthetic.py`) |
| `operations/` | Functions from graphs to a graph: the registry of operations and their phases (`operation.py`), the stated pipeline, its steps and its search space (`pipeline.py`), the shipped operations one per module (`periods.py`, `window.py`, `nodes.py`, `derive.py`, `relations.py`, `edges.py`, `merging.py`, `merge.py`, `difference.py`, `nearby.py`, `consecutive.py`, `neighbourhood.py`) and the rows they share (`rows.py`), forms of storage of set edges (`sets.py`), parameters and settings (`parameters.py`), the segment an operation works on and its carried state (`state.py`), and their kernels (`kernels.py`) |
| `representations/` | The arrays a family of models reads: the registry of representations and their families, structure, sample and encode (`registry.py`), the arrays every variant holds, their carried state and the arrays over a whole variant (`base.py`), one module per shipped representation (`coupling.py`, `events.py`, `neighbors.py`, `walks.py`, `mixed.py`), what they share (`groups.py`) and their kernels (`kernels.py`) |
| `engine/` | Derivation of variants, one engine in segments: derivations, continuations and forks, and the build of a variant in the calling process (`segments.py`), the steps of a pipeline on one segment (`transform.py`), the same on worker processes (`schedule.py`), slices and edits (`edits.py`), writing and stitching segments (`stitch.py`), the physical join tests compare against (`join.py`) |
| `descriptions/` | What the library writes about graphs: `inspect` |
| `registries.py`, `budgets.py` | Registries of named entries; limits of a derivation |
| `infra/` | Code that knows nothing of graphs: files, runtime home, timing, logs, hashing, dense algebra, arrays on the device, the digest of the code (`code.py`), and `processes/` with pools, workers, locks and platform code |

A module imports only its own group, `infra/` and the groups below it; `tests/test_standing.py` holds the order. `infra/` imports nothing else of `graphspec` and nothing of `tuning`.

Domain-free machinery that reads a typed temporal truth graph from staged Parquet or an optional LadybugDB database and derives the graph variants a model search explores. Structure comes from the schema a task file states, documented in [the contract](../docs/contract.md). Mechanism lives in the `graphspec.infra` package: files and arrays, database access and hashing.

Set an explicit absolute `GRAPHSPEC_HOME` before importing the package or launching Python. There is no default home or profile configuration. On Windows, output must resolve to that runtime volume; managed caches and temporary files stay beneath the home. The checkout launcher accepts mandatory `-Home`; see the [storage and loader guide](../docs/loader.md).

| File | Role |
|---|---|
| `schema/schema.py` | `Schema`, the one statement of kinds, edge tables, columns, graph attributes, the model of time with its units and rule for ties, and whether observers are named; the registries of column types and encoders; the one table of type names written to staged tables and its inverse; `Encoding`, the dictionaries of a bank and, once fitted by `fitted` on what the views of the splits trained on show, the states of its encoders. |
| `descriptions/inspect.py` | `inspect` prints the schema of a task file or of a bank. |
| `sources/ladybug.py` | `Database` checks the tables of a database against the schema. `Reader` reads one graph with Cypher anchored on the entity tables. `export` writes staged tables to a database. They go through `graphspec.infra.database`, which handles database access and `copy` import. The public Parquet `Stage` is implemented in `build/output.py`. |
| `schema/references.py` | The sorts of reference, registered: what a task may refer to, how it is stored and how it resolves at the time of the task. |
| `schema/declaration.py` | `EdgeType` and `Registry`, which allocates relation codes, indexes types and fingerprints the declaration; `Registry.from_declaration` rebuilds a registry from a pinned declaration; `registry(task, encoding, several, untasked)`: the relations stated or taken from a column, and every relation a spec can derive; `Registry.grown(entry, added, states)`, the registry with what an entry of a pipeline adds. |
| `schema/added.py` | `Added`, what an operation adds: columns, kinds of node with the tokens of their nodes in a graph, columns placed on kinds, relations, and pairs joined to relations. |
| `schema/time.py` | The models of time, the rules for ties and the choices among several facts, each a registry. |
| `views/visible.py` | `visible` and `allowed`, the one guard that says which facts a task sees, and the rules of what a split may see, registered. |
| `build/forms.py` | The shorthand forms, registered, each reducing rows to facts with intervals. |
| `build/parse.py`, `facts/stored.py`, `facts/check.py` | `parse` turns one graph's rows into `Facts`, the stored facts with their intervals, and `check` checks them. |
| `views/graph.py` | `Graph` is a view as spans over steps, which `verify` checks; `Graph.dense()` expands the spans into the per-step working graph by reference. |
| `views/view.py` | The choices of steps, registered; `sequences` gives the sequences of a graph and `view` shows the facts of a graph at the steps of one sequence as a `Graph`. |
| `views/spans.py` | Interval primitives: encoding rows as spans, expanding spans to rows, fact order, coverage checks. |
| `operations/parameters.py` | Conditional parameters, and the settings of a pipeline generated from a registry and a set of representation groups, with `draw`, `canonical` and `digest`. |
| `operations/operation.py` | `Operation`, the registry `OPERATIONS`, the settings an operation reads its parameters and its splits through, its counts and `listed`. |
| `operations/pipeline.py` | `resolved` checks a stated pipeline and gives its steps; `space`, `entries` and `flattened` give a search the presence and the parameters of every entry of a template under names that stay the same when another entry is added. |
| `operations/periods.py` … `operations/merging.py`, `operations/rows.py` | One module per shipped operation, each registering its operation with its parameters; the rows of facts and observations they select, extend and prune. |
| `representations/` | Model inputs built from a derived segment, stitched over a variant, and their validation, described below. |
| `engine/transform.py` | `derive` runs the steps of a pipeline on one segment in the order stated, checks the schema after each and verifies the segment after the last; `validate` shows every graph of a bank once. |
| `engine/join.py` | `unify` gives the members of a joined group one entity numbering; `joined` concatenates them into one physical graph, the reference the engine is checked against. |
| `engine/segments.py`, `engine/schedule.py`, `engine/edits.py`, `engine/stitch.py`, `operations/state.py` | The one engine, described below. |
| `operations/kernels.py`, `representations/kernels.py` | The numba kernels behind the stages and the representations; `verify`'s is in `views/graph.py`. |

**Truth graph and views.** A bank of version 8 stores one graph as `stored.Facts`, facts with the interval in which each holds; [the contract](../docs/contract.md) lists its arrays. Steps are a view the pipeline chooses by `time/steps` and `time/several`. A view is first shown as stated, a `graph.Shown`, and then encoded, with the encoders fitted for its statement, into a `graph.Graph`, spans over the steps of the view, so an unchanged observation or fact costs one row however long it is shown:

- `kinds` and `entity_token` (the 64-bit hash of `type:key`) per entity, and per step `period`, `step_time`, the time the step shows, and `step_token`, the 64-bit hash of the name of the graph
- observation spans `obs_entity`, `obs_start`, `obs_stop` (exclusive), `obs_values` and `obs_x`: an entity's maximal run of consecutive steps with identical stated values, the values as stated and as encoded; within a step observations are ordered by entity
- fact spans `fact_source`, `fact_target`, `fact_code`, `fact_start`, `fact_stop`, `fact_values`: a maximal run of consecutive steps at which the same fact with the same attributes is present, and `fact_rank`, which with the patches `order_step`, `order_position`, `order_span` reproduces the `seq` order of facts within every step exactly
- removals `removal_source`, `removal_target`, `removal_code`, `removal_step`, `removal_values`: the absent facts, in their `seq` order
- tasks `task_step`, `task_key`, `task_kind`, `task_values`, `task_x`, `task_signals`, the members of their references `member_task`, `member_name`, `member_place`, `member_source`, `member_target`, `member_code`, targets `label_task`, `label_row`, and `graph_values`, the attributes of the graph

`verify` checks a view, in time linear in the spans: steps in range, a period for every step, canonical relation codes, tasks in the order of their steps, observation spans of one entity that neither overlap nor could merge, present facts observed at every step of their span, the nodes that the references of a task name observed at its step where its kind states that an absent node stops the work, absent facts observed at or before theirs, and tokens that match the keys. The encoded values are checked where they are made, against the limit of 16 bits. A bank build refuses two keys of one type that share a token.

The stages work on `graph.Dense`, the per-step expansion: rows reference their features and attributes (`obs_ref`, `fact_ref`) in `x_table` and `value_table` instead of copying them, so a stage that copies or filters rows moves integers. `step_graph` names the graph of the group every step came from, `step_time` and `step_token` travel with the steps through joins, cuts and stitching, and `entity_connected` whether an entity may appear in a later segment.

**Draws.** Every random choice of an operation or a representation is a number drawn by `infra.hashing.unit(seed, salt, *keys)`: the seed the derivation is given, a salt naming the draw, and keys of identity only: the entity's token (`Segment.tokens`), for an edge the tokens of its two ends, ordered where the edge is read symmetrically, and the hash of its relation's name, the graph's name from `step_token`, and for a draw made per step the time from `step_time`. No draw reads a position in an array, an entity number, a step number, a worker, a device or a random generator. A draw that picks one of several candidates takes the candidate with the largest weight plus a Gumbel number drawn from the candidate's own keys, and candidates are ordered by their keys wherever a tie or a cut depends on their order. `infra.hashing.hashed(*keys)` gives the 64-bit identity of a row of keys, as for the events of the walks.

**One engine.** Every variant is derived in segments, and a whole graph is one segment. `segments.opened` prepares a segment: the steps of one graph from a step to another (`edits.sliced`, which keeps the rows of the tables they read), an edit applied to them, its entities numbered by the rank of their number in the derivation, so every sort and tie-break matches the whole, while draws receive the keys of entities and the time and graph of steps. An entity may appear in a later segment when it is connected across the graphs of a group or its graph continues after the segment; only those are carried. `transform.derive` runs the steps on the segment, each with the state its operation carried from the segment before: window copies still open, running degree and activity counts, fact runs, orphan states, derived facts live at the boundary, typed relations still live, the group of the last period; a phase that brings carried entities into a segment widens it (`Segment.widened`). `representations.base.represented` then gives the segment's arrays, with the state of the base (last observations, replay states, counts, the groups of periods and a halo of recent rows for the lookbacks bounded in time: event DAG, boundaries, walks) and of each representation (coupling tails, mixed chains). Every segment's rows are written with the numbers of the derivation and offsets, then stitched into the variant, which is bitwise equal to deriving and representing the whole graph, or the physical join of a group, in one segment.

`segments.variant` runs the segments of a variant in order in the calling process; pieces stay in memory up to 64 MiB and are then written to the variant's files as they come (`stitch.Appender`), so a group's size is bounded by disk, not by memory. When a split has few groups, `schedule.variants` spreads the same work over worker processes: every phase of every step is a stage, run in order on the segments of a group where it carries state, on any segment whose input is ready where it carries none, and part by part where it runs in parts (the filters of `graphspec.edges`, one part per relation), so that `graphspec.derive` computes its steps in parallel across segments. A stage's carried state is small, so it is written beside the spill files after every segment and any idle worker runs the next segment of any stage; only the representations stay on one worker, which appends every segment's arrays to the group's variant. The arrays are equal for any number of workers.

An operation may add kinds of node, relations and columns (see the contract, What an operation adds); the derivation is then given the registry of its pipeline, `pipeline.grown(registry, steps, states)`. `segments.Plan` numbers the nodes of every added kind before the entities of its graphs, and `segments.opened` widens the tables of values of each segment to the registry. Rows of values an operation or an edit adds are marked with no place in the derivation (`carry.x_global`, `carry.value_global` below zero) until `segments.placed` gives them places after every row of the graphs and of the segments before, once the steps of the segment ran and before its representations, in the calling process and on the worker of the representations alike; where a row sits depends on the cuts of segments, what every observation and fact reads does not.

`segments.Derivation` holds a derivation as a value: `begun` states the initial state of every step and representation, `continued` derives from its step to a later one and returns a new derivation that holds it as its parent, and `forked` continues it once per edit. A fork shares the segments and the state of its history and holds only what it derived and what its edits added, so the history of every fork at one step is derived once; `arrays` stitches the variant of a chain of derivations and `derived` its derived graph.

**Relation vocabulary.** `declaration.registry` numbers node types and base relations in the order the schema states them, the relations an edge table takes from a column in sorted order at the place of their table, then declares every relation a spec can derive. A relation stated without direction is symmetric. A directed relation has a canonical and a reverse code; a symmetric relation has one.

A table whose relations are taken from a column is one table with a value. Each value keeps its own code, which facts, references and events carry and which models read through a learned embedding; everything else is the table's: its parameters under `relations/<table>/` and `edges/<table>/`, its projections, its array `attributes_<table>`, and the weights models hold per relation. `Registry.groups` names, for every relation stated, every table of relations taken from a column and every relation of a set edge, the relations it holds; `Registry.coded` their codes, with every derived relation alone; `Registry.grouped` the index of the weights of every code, one per table and direction; `Registry.valued` the tables whose relations carry values, then the relations an operation added with values, `Registry.value_group` the place of each code among them, and `Registry.attributed` the most values a relation holds. `Registry.declared` names the kinds of node and of set edge the bank declares, which per-kind parameters read; `Registry.nodes` adds the kinds a pipeline added. What grows with the number of values of a column is the declaration of the registry, its tables indexed by code, and the embedding tables of relations in models.

| Relation | Symmetric | Meaning |
|---|---|---|
| `<relation>` | as stated | A base relation, from an edge table. |
| `project:<table>:<side>:<A>:<B>` | yes | Entities of types A and B on one side of a relation that share a counterpart at a step. One per relation stated, table of relations taken from a column or relation of a set edge, side and pair of types whose FROM/TO pairs share a counterpart type. |
| `complete:<A>:<B>` | yes | Pairs of completed entities of types A and B at a step. |
| `hop:2`, `hop:3` | yes | Pairs at exactly that shortest distance in the step's graph. |
| `diffusion` | no | Top-k graph-diffusion neighbours. |
| `rewire` | yes | Edges added by curvature or spectral rewiring. |
| `untyped` | no | Every base relation collapsed into one. |
| `temporal` | yes | An entity's link to its own earlier observation. |
| `boundary:first`, `boundary:last`, `boundary:both` | yes | Links from the first, last or only observation or event of an earlier period. |

The markers `non_relation` and `period_context` take codes above every relation. The registry extras record:

- the kind of target, a number or a path, and the series
- whether a kind of task is seen as an event
- the signal width
- the base relations
- whether a step can show several facts about one entity, and the graphs that hold no task

`declaration()` serialises everything and `fingerprint()` hashes it. The bank manifest, the run plan and every trial's `selection.json` pin that hash.

**Pipeline and settings.** A configuration states its pipeline, the operations in order with their parameters, and its settings, a flat dictionary of the rest: forms of storage, steps, joining, how models read kinds and directed relations, the parameters of the representation groups a model reads, and those read when packing; [the contract](../docs/contract.md) states both. `parameters.settings(registry, groups)` generates the settings for one registry and the representation groups a model reads; an operation generates its own parameters. Each parameter is active only under its condition; `nodes/<type>/min_degree`, for example, needs `nodes/<type>/filter` set to degree or total_degree.

- `draw` walks the parameters in order and asks for a value for each active one.
- `canonical` refuses a missing active parameter, an inactive or unknown one, and an out-of-range value.
- `digest` is the SHA-256 of the sorted JSON.
- Every draw of the operations and representations is keyed by the seed the derivation is given, `transform.SEED` for the variants of a bank, with a salt per draw, so changing one parameter never reshuffles the random choices of another.
- `materialized` drops the settings read only when packing: `time/history`, `time/context`, `neighbors/order` and `neighbors/decay`. A cached variant is named by the key the contract states: the schema, the source, the pipeline with the version of every operation, the settings, the groups, what the splits may see, the device, the seed and the versions of the code of views, derivation and representations; no file is hashed.

**Operations.** `transform.derive` applies the steps of a pipeline to a segment in the order it states; each operation reads its parameters through the settings of [the contract](../docs/contract.md), the graph the step before it produced and, where it takes two graphs, the rows of its second graph at the steps of the segment (`state.Second`, which `segments.opened` cuts with the segment). A step that changes nothing returns its input, whose schema is not checked again, and every array a step leaves unchanged is shared with its input; the segment is verified once after the last step; a relation whose edge filters sit at their identity values is left untouched. The seven shipped operations follow, numbered in the order the fixture of 1.2 applied them; a pipeline may state any of them, in any order, and one more than once. Rows marked as settings belong to the settings, not to an operation.

1. Periods

| Parameter | Range | Effect |
|---|---|---|
| `time/periods` | truth or steps, every | truth, offered where the tasks state a grouping, keeps it as the periods; steps, offered where they state none, makes every step a period; every starts a new period every `time/period_length` steps. Periods are runs of equal values and need not increase. |
| `time/period_length` | 1–16 with tasks seen as events, 2–16 without | Steps per period. Offered for every. |

2. Time

| Parameter | Range | Effect |
|---|---|---|
| `time/scheme` | window, state | Which earlier facts a snapshot holds. window: facts stated in the last w steps, or in the current period and the previous p periods. state: every earlier fact. In each scheme a fact leaves when it is removed or stated again, and an observation carries forward over the same span until the entity's next observation. Parallel facts are copied together. |
| `time/unit` | step, period | What the window counts: steps, or the periods of the graph it is given. Offered for window. |
| `time/window` | 1–8 | w. Offered for window counted in steps. |
| `time/past_periods` | 0–3 | p, counted in distinct periods whatever their numbers. Offered for window counted in periods. |
| `time/history` | 1–32 | A setting: steps in each training window; read when packing. |
| `time/context` | 0–3 | A setting: earlier periods replayed before each training window; read when packing. |

3. Nodes

Every unprotected type has its own switch and filter.

| Parameter | Range | Effect |
|---|---|---|
| `graphs/join` | separate, single | A setting, as are the three below. separate transforms every truth graph on its own. single joins truth graphs in groups of `graphs/join_size` onto one timeline: graphs follow each other in time, periods keep increasing across them, labels and splits keep their rows, and entities of connected types with the same key become one entity so graphs meet through the entities they share. A connected entity, and a fact between two connected entities, carries across the boundaries between graphs under every stage's rules; every other entity and fact ends with its own graph. `every` periods restart at each graph. |
| `graphs/join_size` | 2–4096 | How many truth graphs one joined graph holds. Graphs whose rows fall in the same splits are ordered by name, then observer and split, and cut into groups in that order, never by their place in the bank, so a graph-level split never mixes training and validation graphs in one joined graph and reordering the graphs of a bank leaves every group and its order equal. Offered for single. |
| `graphs/connect/<type>` | on, off | Whether entities of the type are unified by key when joining. Offered for single. |
| `graphs/connect_fraction` | 0.01–1, logarithmic | The share of keys that are unified, chosen by a draw keyed by the token of the key alone, so that every graph of the group chooses alike; the rest stay per graph. Offered for single. |
| `nodes/<type>/keep` | on, off | Removes the entities of the type and every fact that touches them. |
| `nodes/referenced` | keep, and remove where every kind of task keeps an absent node | What becomes of a node that a reference names and that the stage would remove at the step of the task: kept as an anchor, or removed. |
| `nodes/untyped` | on, off | A setting: gives every entity one kind in the model inputs. The operations still read types, and features are untouched. Offered with more than one node type. |
| `nodes/<type>/filter` | none, degree, total_degree, core, activity, sample | Removes observations of the type's entities that fail the filter, and the facts that lose an endpoint. Offered while the type is kept. |
| `nodes/<type>/min_degree` | 1–16 | degree: an observation stays if its entity has at least that many stated incident facts among kept types at that step, a self-loop counting once. total_degree: counted over every step so far. |
| `nodes/<type>/core` | 2–16 | core: observations of every core-filtered type are removed together, repeatedly, while their incident facts to still-kept observations at that step, a self-loop counting once, number fewer than their type's threshold. Observations that failed another filter do not count. |
| `nodes/<type>/min_activity` | 2–16 | activity: an entity stays from its n-th observation, counting the copies an earlier window carried forward. |
| `nodes/<type>/sample` | 0.1–0.9 | sample: a draw keyed by the entity's key and the name of its graph keeps that fraction of the type's entities for the whole graph; in a joined group each graph draws for itself. |

With `nodes/referenced` set to keep, an observation that a reference of a task names at the step of the task is never removed; when its type is removed or it fails its type's filter, it stays as an anchor, marked in `obs_anchor`: no present fact touches it and no derivation links it, while removals of relations its entity held earlier still end at its step. Representations built on observations (coupling, events, mixed) include anchors like any other observation.

4. Derived relations

Computed at each step from the present facts of the graph it is given. Projections, completion anchors, hops, diffusion and rewiring read the base relations that graph holds, as undirected pairs: a relation an earlier step removed is not read, and one a later step removes is.

| Parameter | Range | Effect |
|---|---|---|
| `derive/project:<table>:<side>:<A>:<B>` | on, off | Adds that projection. Offered for every projection the bank declares. |
| `derive/project_cap` | 0–64 | Only each counterpart's `cap` most recently started relations of the step take part, counting parallel facts once, relations started at one step ranked by the token of the member; 0 takes all. Offered when a projection is on. |
| `derive/complete/<type>` | on, off | Completes that type: every pair of observed entities of the completed types at a step is linked. Offered for every type. |
| `derive/complete_within/<type>` | on, off | Links completed entities only when they share a neighbour of that type; with several types on, a shared neighbour of any of them is enough. Offered while a relation of the bank joins this type to a completed type. |
| `derive/hops` | 1–3 | Adds `hop:2` up to `hop:k`. Offered where the bank states a relation. |
| `derive/diffusion` | none, ppr, heat | Scores pairs by personalised PageRank α(I−(1−α)T)⁻¹ or the heat kernel exp(−t(I−T)), with T the symmetric-normalised adjacency of the step's observed entities. Each entity links to its highest-scoring non-adjacent entities in its own connected component. Computed per component in float64; the infra layer chooses the device by component size. Offered where the bank states a relation. |
| `derive/diffusion_top_k` | 1–16 | Neighbours per entity. The scores of a component are computed with its entities in the order of their tokens, and equal scores are cut in that order. Offered with diffusion. |
| `derive/ppr_alpha` | 0.05–0.5 | α. Offered with ppr. |
| `derive/heat_time` | 0.5–5 | t. Offered with heat. |
| `derive/rewiring` | none, sdrf, fosr | fosr works inside connected components. Each addition goes to the component with the smallest spectral gap, at the non-edge that minimises the product of degree-scaled second-eigenvector entries of the component's self-looped normalised adjacency. sdrf takes the edge of lowest balanced Forman curvature. It adds an edge between the two neighbourhoods, sampled by softmax of temperature × curvature gain with Gumbel numbers drawn from the keys of the pair's ends, the time and graph of the step and the iteration, and removes the most curved edge above the threshold. Both order the entities of a step by their keys, so ties go to the first by key. Curvature runs in a numba kernel, and fosr reuses a component's decomposition until an edge changes it. Offered where the bank states a relation. |
| `derive/rewiring_budget` | 1–64 | Iterations per step. Offered with rewiring. |
| `derive/sdrf_temperature` | 1–500, log | Sampling temperature. Offered with sdrf. |
| `derive/sdrf_threshold` | 0.1–20, log | Curvature above which an edge is removed. Offered with sdrf. |

A derived pair is stated at every step it holds and gets an absent row at the step after each run ends. Base facts that rewiring removes are deleted at that step.

Derivation stops with an error once a variant's derived facts exceed the budget the infra layer sets from the machine's memory, and the time window stops the same way when the observation or fact copies it would carry exceed that budget, so a spec that completes a large type into a clique, or carries state across a long joined graph, fails its trial instead of exhausting the machine.

5. Relations

| Parameter | Range | Effect |
|---|---|---|
| `relations/<table>/keep` | on, off | Removes a relation stated, every relation of a table taken from a column, or a relation of a set edge. Offered for every relation of the bank. |
| `relations/direction` | inverse, symmetric, directed | A setting: how models read directed relations: both directions with the reverse code, both directions with one code, or the stated direction only. Symmetric relations are always read both ways. Offered where the bank states a relation. |

6. Edge filters

Every relation stated, table of relations taken from a column, relation of a set edge and derived relation has its own filter under `edges/<table>/`. A base filter is offered for every relation of the bank; a derived relation's while the entry of `graphspec.derive` without a label can produce it: its projection; both of its completed types, with no anchor type on or an anchor type that relations of the bank join to both; hops of at least its distance; diffusion; or rewiring. A pipeline without that entry has no filter of a derived relation. Drop and persistence treat an edge of each value as an edge of its own; a cap counts the edges of every value of a table together.

| Parameter | Range | Effect |
|---|---|---|
| `edges/direction` | inverse, symmetric, directed | How the filters read directed relations, as `relations/direction` says models do. Offered where the bank states a relation. |
| `edges/<table>/drop` | 0–0.9 | A draw keyed by the keys of a pair's ends, its relation's name and its graph's name removes that fraction of entity pairs, with all their rows. |
| `edges/<table>/persistence` | 1–8 | A present fact is kept from the p-th consecutive step it is stated; facts stated twice at one step share a position. |
| `edges/<table>/cap` | 0–64 | At most c neighbours per entity and step, with parallel facts counted once; 0 keeps all. A directed reading caps destinations, inverse caps destinations and sources separately, and symmetric caps each entity with a pair counted once. A pair stays only if it ranks within c at every capped end. |
| `edges/<table>/cap_order` | recent, random, largest:&lt;attribute&gt; | Which facts a cap keeps: latest run start, a draw keyed as the drop's, or the largest value of one of the table's attributes, then the latest run start. Pairs still equal are ranked by the tokens of their ends, ordered where the pair is read symmetrically, then by the hash of their relation's name, never by row or entity number. Offered when the cap is on. |

Relations read symmetrically count their two directions as one pair in every edge filter.

Whenever a stage removes present facts, it also removes absent rows left without any earlier kept present row. The event history therefore never removes a relation the variant never stated.

7. Relation merging

| Parameter | Range | Effect |
|---|---|---|
| `relations/types` | typed, untyped | untyped replaces every base relation with `untyped` and drops attributes. Its absent row stays only when no underlying relation between the pair is still present. Offered where the bank states a relation. |
| `relations/parallel` | keep, merge | merge keeps the last of identical facts stated at one step, last in the order the rule for ties, then the time they hold from, give facts about one edge; facts of relations made untyped carry no values, so which of them is kept changes nothing. Offered where the bank states a relation. |

The operations added after the seven follow. Each states what it requires of the schema and stops naming what is missing.

8. Merge of graphs on keys, `graphspec.merge`, which takes two graphs

The entry names its second graph, a graph of the bank shown at the steps of the graph it changes (see the contract, Second graphs). At every step the merge holds the observations and the facts of both graphs, their entities met by kind and key. Where both hold an observation of one entity, or rows about one edge (the same relation between the same two entities, in either order for a relation without direction), the step keeps those of one graph. Facts whose ends are not both observed, other than as anchors, are dropped, and absent rows left without an earlier present row are removed, as every stage that removes facts does. It carries the state of that removal and nothing else. It draws nothing.

| Parameter | Range | Effect |
|---|---|---|
| `merge/prefer` | first, second | Which graph keeps its rows where both hold an observation of an entity or rows about an edge at one step: the graph changed, or the second. |

9. Difference of two graphs, `graphspec.difference`, which takes two graphs

The entry names its second graph as a merge does. At every step the difference removes the present facts of the graph that the second graph also holds present at that step: the same relation between the same two entities, met by kind and key, in either order for a relation without direction. Observations are kept, and absent rows left without an earlier present row are removed. It carries the state of that removal and nothing else. It draws nothing.

| Parameter | Range | Effect |
|---|---|---|
| `difference/match` | keys, values | keys: an edge the second graph holds at the step is removed whatever its values; values: only where the second graph's edge carries the same values, compared as stated, bit for bit, an edge without values matching one without values. |

10. Edges computed from the columns of two nodes, `graphspec.nearby`

Adds a relation named after its entry, from nodes of a first kind to nodes of a second, whose edges hold their distance in a column it adds, `<entry>.distance`, a number encoded by the encoder the entry states and fitted as every added column is. It requires two kinds of node or of set edge that state a column of numbers both state, and stops naming the bank where none are; the distance between two nodes is the Euclidean distance between the encoded values of the columns of numbers their kinds state alike, never of columns an operation added, and stops naming the kinds where the encoders fitted for the statement give none of those columns a width. At every step it reads the nodes of the two kinds observed at the step, other than as anchors, and joins no node missing a value in one of those columns. An edge is stated at every step it is found, as stated edges are, with no absent row when it is no longer found. Under a threshold the relation has no direction and holds each pair once, the lower entity first where both ends are of one kind; under nearest it is directed, from each node to its own nearest. It carries no state from segment to segment and draws nothing.

| Parameter | Range | Effect |
|---|---|---|
| `nearby/kinds` | `<first>:<second>`, every two kinds that qualify | The kinds it joins, from the first to the second. |
| `nearby/rule` | threshold, nearest | threshold: every two nodes within the distance; nearest: every node of the first kind to its nearest of the second. |
| `nearby/threshold` | 0.01–100, logarithmic | The distance. Offered for threshold. |
| `nearby/k` | 1–16 | How many nearest each node joins, fewer where fewer are observed; candidates at equal distances are taken in the order of their tokens. Offered for nearest. |
| `nearby/encoder` | the encoders of numbers, no default | The encoder of the distance. |

11. Edges between consecutive rows of a group, `graphspec.consecutive`

Fills a relation the schema states without columns from a kind of node to itself, and stops naming the bank where none is. The rows are the nodes of that kind. A row joins a group at the first step it is seen in it: with the whole graph as one group, at its first observation other than as an anchor; with a group per counterpart, at the first step at which a present edge of the grouping relation links it, on the side stated, to that counterpart, so that a row with several counterparts is in several groups. In a group, rows follow one another in the order of the time they joined, then of their tokens; each row is joined to the row before it, from the earlier to the later, at the step the later joined, where the earlier is observed at that step other than as an anchor and joined at most the gap before; a longer gap, or another graph of a joined group, starts the group again. Edges are stated at that step only, without values; a pipeline that states `graphspec.window` after it carries them on. It carries the rows that joined each group and the last row of each group from segment to segment. It draws nothing.

| Parameter | Range | Effect |
|---|---|---|
| `consecutive/pairing` | the relations that qualify, alone or with a grouping | `<relation>`: the rows of the whole graph form one group; `<relation>:<grouping>:<side>`: the rows are grouped by their counterparts over the relation `<grouping>`, on which they are the sources or the targets. |
| `consecutive/gap` | 1–10⁹, logarithmic, in the unit of time of the graph | The longest time between two consecutive rows that are joined. |

12. Neighbourhood of nodes to a number of hops, at a time, `graphspec.neighbourhood`

At every step it keeps the part of the graph within a number of hops of its nodes at that step, and removes the rest: the observations of the nodes reached, the present facts between two nodes reached, and the absent facts whose ends were kept at or before their step. Its nodes at a step are those the tasks of the step reference, observed at it, and, where a kind is stated, every node of the kind observed at it; a node a task references is therefore never removed at the step of the task, and a step without such nodes keeps nothing. Hops follow the present facts of the step in either direction, whatever their relation; it counts paths, and under the forms of storage of set edges counts them as stored, as `graphspec.derive` does. It carries the nodes it kept, for the absent facts of later segments, and the state of removing absent rows left without an earlier present row. It draws nothing. `neighbourhood.within(seeds, source, target, hops)` gives the keys reached from seeds over pairs of keys.

| Parameter | Range | Effect |
|---|---|---|
| `neighbourhood/hops` | 0–3 | How many hops from its nodes; 0 keeps its nodes alone. |
| `neighbourhood/around` | references, or a kind of node | Its nodes: those the tasks of the step reference, and every node of the kind where one is stated. |

**Representations.** `groups.build` always produces the base arrays and adds the groups a model reads.

Base arrays:

- snapshots: `x_ref` per observation into the feature table `x_table`, the `obs_x` of the views of the variant's graphs in order, widened to the registry of the pipeline, followed by the rows of values its edits and operations added, segment by segment; `entity`, `step`, `obs_period` and `snapshot_offsets` per observation; `observation_ids` and `observation_offsets` per entity; `entity_kind`, all zero when nodes are untyped; relation edges `spatial` as source observation, target observation and code, with `spatial_step`, `spatial_attribute_row` and `attributes_<table>`, one array of values for every table whose relations carry values
- rows, one per task of a kind that states a target: `row_task`, `row_step`, `row_kind`, `row_index`, `key`, `signals`
- references of the rows: `member_row`, `member_name`, `member_entity`, `member_observed`, `member_place`, one entry per node a reference names, an edge giving its two ends, ordered by row, then by the position of the name among the references of the schema, then by place; `member_place` is twice the place of the member plus the end of its edge for a sort that keeps order, and 0 for any other; `whole_row` and `whole_name` for references to the graph itself. Nothing is kept by position: no family reads a reference by its place in the kind.
- tasks: `task`, the features of every task and the attributes of its graph, and for the tasks seen as events `event_first`, `event_task` and `event_step`
- `events`: one row per observation change, relation change or task seen as an event. Observations and facts of step s sit at tick 2s and the tasks of step s at tick 2s+1; the row of a task holds its place in `task`.
- periods: `period`, and the index `period_starts`, `period_ends`, `period_numbers`, `period_membership`, `period_observation_keys`; `affinity_keys` and `affinity_counts`, per task seen as an event ordered by its first node and tick, the counts by kind of the tasks seen as events that node was first of up to and including it; `affinity_steps`, per step, the counts by kind of the tasks seen as events before it

| Group | Parameters | Arrays |
|---|---|---|
| `coupling` | `coupling/scheme` ordinal, lags, window, all; `coupling/unit` observation, step, for ordinal and lags; `coupling/max_lag` 2–64, for lags; `coupling/lag_window` 1–512, for lags by observation; `coupling/window` 2–128 and `coupling/cap` 1–128, for window; `coupling/periods` 0–3; `coupling/period_window` 2–128, for periods above 0 | `coupling` links an observation to earlier observations of its entity. ordinal takes the previous one. lags takes those 1, 2, 4, … up to and including the exact `max_lag` back, within `lag_window` steps. window takes every earlier one less than `window` steps back, at most the latest `cap`; an entity has one observation per step, so the cap never meets a tie. all takes every earlier one. With unit step, ordinal and lags count steps and need an observation exactly that far back. `coupling_boundaries` links from the entity's first and last observations in each of the previous `periods` periods, less than `period_window` steps back. |
| `events` | `events/kinds` any subset of observation, relation and, with tasks seen as events, task; `events/link` next, lags, window; `events/max_lag` 2–16, for lags; `events/window` 1–1024; `events/cap` 0–64; `events/periods` 0–3; `events/period_window` 2–128, for periods above 0 | `event_dag` links each member event, once per earlier event, from its participants' earlier tick groups within `window` ticks. next takes the previous group, lags the groups 1, 2, 4, … up to and including the exact `max_lag` back, and window every earlier group. With `cap` above 0, only `cap` events of each group link: the events of one participant at one tick are ordered by `event_identity` and the last `cap` kept, so the cut never depends on the numbering of entities or the order of facts. `event_member` marks the chosen kinds. `event_boundaries` links from the first and last tick groups of each participant's previous `periods` periods, less than `period_window` steps back. |
| `neighbors` | `neighbors/kinds` relation and, with tasks seen as events, task or both; `neighbors/removals` on, off, when relation events are included; `neighbors/order` recent, uniform, decay; `neighbors/decay` 0.001–1, log, for decay | `neighbor_ids` and `neighbor_offsets` index each entity's relation events, with or without removals, and the events of tasks. An entity's events are listed by tick, then by `event_identity`, the identity the walks read. When packing, a model takes its neighbour count from before each cutoff: the latest, cut by identity where ticks tie, a uniform sample, or a sample weighted by exp(−decay × elapsed steps), drawn by Gumbel top-k with `hashing.unit(seed, "tuning.batches/neighbors", ...)` keyed by the entity's token, the graph and time of the cutoff's step (`step_token`, `step_time`) and its half, and the candidate event's identity, so no sample depends on the numbering of entities and events or on the place of a graph in a join. Candidates start at the packed window's first step. |
| `walks` | `neighbors/kinds`, `neighbors/removals`; `walks/bias` 0–1; `walks/window` 1–512; `walks/fanout` 1–128 | Builds the neighbour index as above. `walks` holds 32 walks of 4 hops from the node of every member of the references of each row, over events within `window` steps of it. Each hop picks one of the current entity's latest `fanout` events strictly before the current time, weighted by exp(−bias × steps before the current time). Each walk is keyed by the key of its root, the time and graph of its row's step and its sample; each hop takes the candidate with the largest weight plus a Gumbel number drawn from the walk's key, the hop and the candidate event's identity, the hash of its kind, the keys of its ends, its relation or task, its state and the time and graph of its step. Candidates are ordered by tick and identity, so no walk depends on other predictions, on the order of references or on the numbering of entities and events. `period_walks` does the same inside each of the 3 previous periods, bounded by the period instead of `window`. |
| `mixed` | `mixed/changes` on, off; `mixed/trace` 0–8; `mixed/bundles` on, off | `mixed_*` is a graph of derived nodes: one per observation linked to the entity's previous one, change nodes holding feature deltas, trace nodes averaging every `trace` observations of an entity, bundle nodes averaging each step, and one node per task seen as an event. |

**Causality.** Every stage and representation reads only steps at or before the one it writes, and the stages never read labels. Truncating a truth graph after any step and transforming it gives a prefix of the transformed full graph, apart from the period-index rows of the last, unfinished period.
