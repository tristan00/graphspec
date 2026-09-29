**Graph spec core**

Domain-free machinery that reads a typed temporal truth graph from staged Parquet or an optional LadybugDB database and derives the graph variants a model search explores. Structure comes from the schema a task file states, documented in [the contract](../docs/contract.md). Mechanism lives in the `graphspec.infra` package: files and arrays, database access, hashing and the feature encoder.

Set an explicit absolute `GRAPHSPEC_HOME` before importing the package or launching Python. There is no default home or profile configuration. On Windows, output must resolve to that runtime volume; managed caches and temporary files stay beneath the home. The checkout launcher accepts mandatory `-Home`; see the [storage and loader guide](../docs/loader.md).

| File | Role |
|---|---|
| `schema.py` | `Schema`, the one statement of kinds, edge tables, columns, graph attributes, the model of time with its units and rule for ties, and whether observers are named; the registries of column types and encoders; `Encoding`, the dictionaries of a bank and, once fitted by `fitted` on what the views of the splits trained on show, the states of its encoders. |
| `inspect.py` | `inspect` prints the schema of a task file or of a bank. |
| `graphdb.py` | `Database` checks the tables of a database against the schema. `Reader` reads one graph with Cypher anchored on the entity tables. Both go through `graphspec.infra.database`, which handles database access and `copy` import. The public Parquet `Stage` is implemented in `graphspec.infra.staging`. |
| `references.py` | The sorts of reference, registered: what a task may refer to, how it is stored and how it resolves at the time of the task. |
| `vocabulary.py` | `registry(task, encoding, several, untasked)`: the relations stated or taken from a column, and every relation a spec can derive. |
| `declarations.py`, `registry.py` | `EdgeType` and `Registry`, which allocates relation codes, indexes types and fingerprints the declaration. `Registry.from_declaration` rebuilds a registry from a pinned declaration. |
| `clock.py` | The models of time, the rules for ties and the choices among several facts, each a registry; `visible` and `allowed`, the one guard that says which facts a task sees. |
| `forms.py` | The shorthand forms, registered, each reducing rows to facts with intervals. |
| `truth.py` | `parse` turns one graph's rows into `Facts`, the stored facts with their intervals, and `check` checks them; `Graph` is a view as spans over steps, which `verify` checks; `Graph.dense()` expands the spans into the per-step working graph by reference. |
| `view.py` | The choices of steps and the rules of what a split may see, each registered; `sequences` gives the sequences of a graph and `view` shows the facts of a graph at the steps of one sequence as a `Graph`. |
| `spans.py` | Interval primitives: encoding rows as spans, expanding spans to rows, fact order, coverage checks. |
| `spec.py` | The spec space: conditional parameters generated from a registry and a set of representation groups, with `draw`, `canonical` and `digest`. |
| `operators.py` | One function per graph stage. |
| `representations.py` | Model inputs built from a derived graph, and their validation. |
| `transform.py` | `derive` runs the stages and verifies the graph after each; `materialize` adds the representations. |
| `join.py` | `unify` gives the members of a joined group one entity numbering; `joined` concatenates them into one physical graph, the reference the segment engine is checked against. |
| `segments.py`, `carry.py` | The segment engine that builds a joined group graph by graph, described below. |
| `kernels.py` | The numba kernels behind the stages, the representations and `verify`. |

**Truth graph and views.** A bank of version 8 stores one graph as `truth.Facts`, facts with the interval in which each holds; [the contract](../docs/contract.md) lists its arrays. Steps are a view the pipeline chooses by `time/steps` and `time/several`. A view is first shown as stated, a `truth.Shown`, and then encoded, with the encoders fitted for its statement, into a `truth.Graph`, spans over the steps of the view, so an unchanged observation or fact costs one row however long it is shown:

- `kinds` and `entity_token` (the 64-bit hash of `type:key`) per entity, and `period` per step
- observation spans `obs_entity`, `obs_start`, `obs_stop` (exclusive), `obs_values` and `obs_x`: an entity's maximal run of consecutive steps with identical stated values, the values as stated and as encoded; within a step observations are ordered by entity
- fact spans `fact_source`, `fact_target`, `fact_code`, `fact_start`, `fact_stop`, `fact_values`: a maximal run of consecutive steps at which the same fact with the same attributes is present, and `fact_rank`, which with the patches `order_step`, `order_position`, `order_span` reproduces the `seq` order of facts within every step exactly
- removals `removal_source`, `removal_target`, `removal_code`, `removal_step`, `removal_values`: the absent facts, in their `seq` order
- tasks `task_step`, `task_key`, `task_kind`, `task_values`, `task_x`, `task_signals`, the members of their references `member_task`, `member_name`, `member_place`, `member_source`, `member_target`, `member_code`, targets `label_task`, `label_row`, and `graph_values`, the attributes of the graph

`verify` checks a view, in time linear in the spans: steps in range, a period for every step, canonical relation codes, tasks in the order of their steps, observation spans of one entity that neither overlap nor could merge, present facts observed at every step of their span, the nodes that the references of a task name observed at its step where its kind states that an absent node stops the work, absent facts observed at or before theirs, and tokens that match the keys. The encoded values are checked where they are made, against the limit of 16 bits. A bank build refuses two keys of one type that share a token.

The stages work on `truth.Dense`, the per-step expansion: rows reference their features and attributes (`obs_ref`, `fact_ref`) in `x_table` and `value_table` instead of copying them, so a stage that copies or filters rows moves integers. `step_graph` and `entity_connected` carry the joined timeline (all zeros for a single graph).

**Joined groups.** A joined group is never concatenated. `segments.build` transforms and represents one member graph at a time, as a segment of the group's timeline: its entities are numbered by the rank of their group id, so every sort and tie-break matches the whole group, and hashes and random streams receive group ids and group steps. Each stage and representation that looks back in time receives the state earlier graphs left it (`carry.Segment.state`): window copies still open, running degree and activity counts, fact runs, orphan states, derived facts live at the boundary, typed relations still live, last observations, replay states, coupling tails and mixed chains. Lookbacks bounded in time (event DAG, boundaries, walks) read a halo of the recent rows instead. Every segment's rows are written with group ids and offsets, then stitched into the group's variant, which is bitwise equal to transforming and representing the physical join. When a split has few groups, `segments.build_parallel` spreads its groups over the workers, and the groups advance side by side. Each stage that carries state runs its segments in order: the time window, the node filters, the derived facts live at the boundary, the edge filters split by relation over up to three shards balanced by row count, relation merging, and the representations. A stage's carried state is small, so it is written beside the spill files after every segment and any idle worker runs the next segment of any stage; only the representations stay on one worker, which appends every segment's arrays straight into the group's variant files. Work without carried state runs wherever a worker is free: expanding graphs from spans, the per-step derivation, and assembling derived facts with the relation filter and the stateless edge drop. Stages hand segments to each other through one spill file each. The variant is verified once per segment after relation merging, and stitching only adds the group-level indexes. Budgets apply per segment, so a group's size is bounded by disk, not by the graph budget.

**Relation vocabulary.** `vocabulary.registry` numbers node types and base relations in the order the schema states them, the relations an edge table takes from a column in sorted order at the place of their table, then declares every relation a spec can derive. A relation stated without direction is symmetric. A directed relation has a canonical and a reverse code; a symmetric relation has one.

A table whose relations are taken from a column is one table with a value. Each value keeps its own code, which facts, references and events carry and which models read through a learned embedding; everything else is the table's: its parameters under `relations/<table>/` and `edges/<table>/`, its projections, its array `attributes_<table>`, and the weights models hold per relation. `Registry.groups` names, for every relation stated, every table of relations taken from a column and every relation of a set edge, the relations it holds; `Registry.coded` their codes, with every derived relation alone; `Registry.grouped` the index of the weights of every code, one per table and direction; `Registry.valued` the tables whose relations carry values, and `Registry.value_group` the place of each code among them. What grows with the number of values of a column is the declaration of the registry, its tables indexed by code, and the embedding tables of relations in models.

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

**Spec.** A spec is a flat dictionary of graph-transform parameters. `spec.space(registry, groups)` generates the parameters for one registry and the representation groups a model reads. Each parameter is active only under its condition; `nodes/<type>/min_degree`, for example, needs `nodes/<type>/filter` set to degree or total_degree.

- `draw` walks the parameters in order and asks for a value for each active one.
- `canonical` refuses a missing active parameter, an inactive or unknown one, and an out-of-range value.
- `digest` is the SHA-256 of the sorted JSON.
- Every hashed or sampled choice in the stages and representations uses one fixed seed, `transform.SEED`, so changing one parameter never reshuffles the random choices of another.
- `materialized` drops the parameters read only when packing: `time/history`, `time/context`, `neighbors/order` and `neighbors/decay`. Its digest, with the groups, the device and what the splits may see, names a cached variant inside the folder of its bank, which is keyed by the fingerprint of the manifest and the digest of the sources of `graphspec` and `tuning`.

**Stages.** `transform.derive` applies seven stages in a fixed order. Each stage reads the graph the stage before it produced. A stage that returns its input unchanged is not verified again, and a relation whose edge filters sit at their identity values is left untouched.

1. Periods

| Parameter | Range | Effect |
|---|---|---|
| `time/periods` | truth or steps, every | truth, offered where the tasks state a grouping, keeps it as the periods; steps, offered where they state none, makes every step a period; every starts a new period every `time/period_length` steps. Periods are runs of equal values and need not increase. |
| `time/period_length` | 1–16 with tasks seen as events, 2–16 without | Steps per period. Offered for every. |

2. Time

| Parameter | Range | Effect |
|---|---|---|
| `time/scheme` | window, state | Which earlier facts a snapshot holds. window: facts stated in the last w steps, or in the current period and the previous p periods. state: every earlier fact. In each scheme a fact leaves when it is removed or stated again, and an observation carries forward over the same span until the entity's next observation. Parallel facts are copied together. |
| `time/unit` | step, period | What the window counts. Offered for window when periods span more than one step: truth periods, or every with a length above 1. |
| `time/window` | 1–8 | w. Offered for window counted in steps. |
| `time/past_periods` | 0–3 | p, counted in distinct periods whatever their numbers. Offered for window counted in periods. |
| `time/history` | 1–32 | Steps in each training window; read when packing. |
| `time/context` | 0–3 | Earlier periods replayed before each training window; read when packing. |

3. Nodes

Every unprotected type has its own switch and filter.

| Parameter | Range | Effect |
|---|---|---|
| `graphs/join` | separate, single | separate transforms every truth graph on its own. single joins truth graphs in groups of `graphs/join_size` onto one timeline: graphs follow each other in time, periods keep increasing across them, labels and splits keep their rows, and entities of connected types with the same key become one entity so graphs meet through the entities they share. A connected entity, and a fact between two connected entities, carries across the boundaries between graphs under every stage's rules; every other entity and fact ends with its own graph. `every` periods restart at each graph. |
| `graphs/join_size` | 2–4096 | How many truth graphs one joined graph holds. Graphs are grouped in bank order among graphs whose rows fall in the same splits, so a graph-level split never mixes training and validation graphs in one joined graph. Offered for single. |
| `graphs/connect/<type>` | on, off | Whether entities of the type are unified by key when joining. Offered for single. |
| `graphs/connect_fraction` | 0.01–1, logarithmic | The share of keys that are unified, chosen by a hash of the key; the rest stay per graph. Offered for single. |
| `nodes/<type>/keep` | on, off | Removes the entities of the type and every fact that touches them. |
| `nodes/referenced` | keep, and remove where every kind of task keeps an absent node | What becomes of a node that a reference names and that the stage would remove at the step of the task: kept as an anchor, or removed. |
| `nodes/untyped` | on, off | Gives every entity one kind in the model inputs. The stages still read types, and features are untouched. Offered with more than one node type. |
| `nodes/<type>/filter` | none, degree, total_degree, core, activity, sample | Removes observations of the type's entities that fail the filter, and the facts that lose an endpoint. Offered while the type is kept. |
| `nodes/<type>/min_degree` | 1–16 | degree: an observation stays if its entity has at least that many stated incident facts among kept types at that step, a self-loop counting once. total_degree: counted over every step so far. |
| `nodes/<type>/core` | 2–16 | core: observations of every core-filtered type are removed together, repeatedly, while their incident facts to still-kept observations at that step, a self-loop counting once, number fewer than their type's threshold. Observations that failed another filter do not count. |
| `nodes/<type>/min_activity` | 2–16 | activity: an entity stays from its n-th observation, counting the copies stage 2 carried forward. |
| `nodes/<type>/sample` | 0.1–0.9 | sample: a seeded hash keeps that fraction of the type's entities for the whole graph. |

With `nodes/referenced` set to keep, an observation that a reference of a task names at the step of the task is never removed; when its type is removed or it fails its type's filter, it stays as an anchor, marked in `obs_anchor`: no present fact touches it and no derivation links it, while removals of relations its entity held earlier still end at its step. Representations built on observations (coupling, events, mixed) include anchors like any other observation.

4. Derived relations

Computed at each step from the present facts that survive stage 3. Completion anchors, hops, diffusion and rewiring read the base relations that `relations/<table>/keep` keeps, as undirected pairs. A projection reads its own relations even when they are removed.

| Parameter | Range | Effect |
|---|---|---|
| `derive/project:<table>:<side>:<A>:<B>` | on, off | Adds that projection. Offered while A, B and a counterpart type their FROM/TO pairs share are kept. |
| `derive/project_cap` | 0–64 | Only each counterpart's `cap` most recently started relations of the step take part, counting parallel facts once; 0 takes all. Offered when a projection is on. |
| `derive/complete/<type>` | on, off | Completes that type: every pair of observed entities of the completed types at a step is linked. Offered while the type is kept. |
| `derive/complete_within/<type>` | on, off | Links completed entities only when they share a neighbour of that type; with several types on, a shared neighbour of any of them is enough. Offered while this type is kept and a kept base relation joins it to a completed type. |
| `derive/hops` | 1–3 | Adds `hop:2` up to `hop:k`. Offered while a base relation is kept. |
| `derive/diffusion` | none, ppr, heat | Scores pairs by personalised PageRank α(I−(1−α)T)⁻¹ or the heat kernel exp(−t(I−T)), with T the symmetric-normalised adjacency of the step's observed entities. Each entity links to its highest-scoring non-adjacent entities in its own connected component. Computed per component in float64; the infra layer chooses the device by component size. Offered while a base relation is kept. |
| `derive/diffusion_top_k` | 1–16 | Neighbours per entity. Offered with diffusion. |
| `derive/ppr_alpha` | 0.05–0.5 | α. Offered with ppr. |
| `derive/heat_time` | 0.5–5 | t. Offered with heat. |
| `derive/rewiring` | none, sdrf, fosr | fosr works inside connected components. Each addition goes to the component with the smallest spectral gap, at the non-edge that minimises the product of degree-scaled second-eigenvector entries of the component's self-looped normalised adjacency. sdrf takes the edge of lowest balanced Forman curvature. It adds an edge between the two neighbourhoods, sampled by softmax of temperature × curvature gain, and removes the most curved edge above the threshold. Curvature runs in a numba kernel, and fosr reuses a component's decomposition until an edge changes it. Offered while a base relation is kept. |
| `derive/rewiring_budget` | 1–64 | Iterations per step. Offered with rewiring. |
| `derive/sdrf_temperature` | 1–500, log | Sampling temperature. Offered with sdrf. |
| `derive/sdrf_threshold` | 0.1–20, log | Curvature above which an edge is removed. Offered with sdrf. |

A derived pair is stated at every step it holds and gets an absent row at the step after each run ends. Base facts that rewiring removes are deleted at that step.

Derivation stops with an error once a variant's derived facts exceed the budget the infra layer sets from the machine's memory, and the time window stops the same way when the observation or fact copies it would carry exceed that budget, so a spec that completes a large type into a clique, or carries state across a long joined graph, fails its trial instead of exhausting the machine.

5. Relations

| Parameter | Range | Effect |
|---|---|---|
| `relations/<table>/keep` | on, off | Removes a relation stated, every relation of a table taken from a column, or a relation of a set edge. Offered while both types of one of its FROM/TO pairs are kept. |
| `relations/direction` | inverse, symmetric, directed | How directed relations are read: both directions with the reverse code, both directions with one code, or the stated direction only. Symmetric relations are always read both ways. Offered while a base relation is kept. |

6. Edge filters

Every relation stated, table of relations taken from a column, relation of a set edge and derived relation has its own filter under `edges/<table>/`. A base filter is offered while its relations are kept; a derived relation's while a derivation can produce it: its projection; both of its completed types, with no anchor type on or an anchor type that kept relations join to both; hops of at least its distance; diffusion; or rewiring. Drop and persistence treat an edge of each value as an edge of its own; a cap counts the edges of every value of a table together.

| Parameter | Range | Effect |
|---|---|---|
| `edges/<table>/drop` | 0–0.9 | A seeded hash removes that fraction of entity pairs, with all their rows. |
| `edges/<table>/persistence` | 1–8 | A present fact is kept from the p-th consecutive step it is stated; facts stated twice at one step share a position. |
| `edges/<table>/cap` | 0–64 | At most c neighbours per entity and step, with parallel facts counted once; 0 keeps all. A directed reading caps destinations, inverse caps destinations and sources separately, and symmetric caps each entity with a pair counted once. A pair stays only if it ranks within c at every capped end. |
| `edges/<table>/cap_order` | recent, random, largest:&lt;attribute&gt; | Which facts a cap keeps: latest run start, a seeded hash, or the largest value of one of the table's attributes. Offered when the cap is on. |

Relations read symmetrically count their two directions as one pair in every edge filter.

Whenever a stage removes present facts, it also removes absent rows left without any earlier kept present row. The event history therefore never removes a relation the variant never stated.

7. Relation merging

| Parameter | Range | Effect |
|---|---|---|
| `relations/types` | typed, untyped | untyped replaces every base relation with `untyped` and drops attributes. Its absent row stays only when no underlying relation between the pair is still present. Offered while a base relation is kept. |
| `relations/parallel` | keep, merge | merge keeps the last of identical facts stated at one step. Offered while a base relation is kept. |

**Representations.** `representations.build` always produces the base arrays and adds the groups a model reads.

Base arrays:

- snapshots: `x_ref` per observation into the feature table `x_table`, which for separate graphs is a copy of the `obs_x` file of its view when no stage changed it; `entity`, `step`, `obs_period` and `snapshot_offsets` per observation; `observation_ids` and `observation_offsets` per entity; `entity_kind`, all zero when nodes are untyped; relation edges `spatial` as source observation, target observation and code, with `spatial_step`, `spatial_attribute_row` and `attributes_<table>`, one array of values for every table whose relations carry values
- rows, one per task of a kind that states a target: `row_task`, `row_step`, `row_kind`, `row_index`, `key`, `signals`
- references of the rows: `member_row`, `member_name`, `member_entity`, `member_observed`, `member_place`, one entry per node a reference names, an edge giving its two ends, ordered by row, then by the position of the name among the references of the schema, then by place; `member_place` is twice the place of the member plus the end of its edge for a sort that keeps order, and 0 for any other; `whole_row` and `whole_name` for references to the graph itself. Nothing is kept by position: no family reads a reference by its place in the kind.
- tasks: `task`, the features of every task and the attributes of its graph, and for the tasks seen as events `event_first`, `event_task` and `event_step`
- `events`: one row per observation change, relation change or task seen as an event. Observations and facts of step s sit at tick 2s and the tasks of step s at tick 2s+1; the row of a task holds its place in `task`.
- periods: `period`, and the index `period_starts`, `period_ends`, `period_numbers`, `period_membership`, `period_observation_keys`; `affinity_keys` and `affinity_counts`, per task seen as an event ordered by its first node and tick, the counts by kind of the tasks seen as events that node was first of up to and including it; `affinity_steps`, per step, the counts by kind of the tasks seen as events before it

| Group | Parameters | Arrays |
|---|---|---|
| `coupling` | `coupling/scheme` ordinal, lags, window, all; `coupling/unit` observation, step, for ordinal and lags; `coupling/max_lag` 2–64, for lags; `coupling/lag_window` 1–512, for lags by observation; `coupling/window` 2–128 and `coupling/cap` 1–128, for window; `coupling/periods` 0–3; `coupling/period_window` 2–128, for periods above 0 | `coupling` links an observation to earlier observations of its entity. ordinal takes the previous one. lags takes those 1, 2, 4, … up to and including the exact `max_lag` back, within `lag_window` steps. window takes every earlier one less than `window` steps back, at most the latest `cap`. all takes every earlier one. With unit step, ordinal and lags count steps and need an observation exactly that far back. `coupling_boundaries` links from the entity's first and last observations in each of the previous `periods` periods, less than `period_window` steps back. |
| `events` | `events/kinds` any subset of observation, relation and, with tasks seen as events, task; `events/link` next, lags, window; `events/max_lag` 2–16, for lags; `events/window` 1–1024; `events/cap` 0–64; `events/periods` 0–3; `events/period_window` 2–128, for periods above 0 | `event_dag` links each member event, once per earlier event, from its participants' earlier tick groups within `window` ticks. next takes the previous group, lags the groups 1, 2, 4, … up to and including the exact `max_lag` back, and window every earlier group. With `cap` above 0, only the last `cap` events of each group link. `event_member` marks the chosen kinds. `event_boundaries` links from the first and last tick groups of each participant's previous `periods` periods, less than `period_window` steps back. |
| `neighbors` | `neighbors/kinds` relation and, with tasks seen as events, task or both; `neighbors/removals` on, off, when relation events are included; `neighbors/order` recent, uniform, decay; `neighbors/decay` 0.001–1, log, for decay | `neighbor_ids` and `neighbor_offsets` index each entity's relation events, with or without removals, and the events of tasks. When packing, a model takes its neighbour count from before each cutoff: the latest, a uniform sample, or a sample weighted by exp(−decay × elapsed steps), drawn by Gumbel top-k with keys hashed from the entity, the cutoff and the candidate event. Candidates start at the packed window's first step. |
| `walks` | `neighbors/kinds`, `neighbors/removals`; `walks/bias` 0–1; `walks/window` 1–512; `walks/fanout` 1–128 | Builds the neighbour index as above. `walks` holds 32 walks of 4 hops from the node of every member of the references of each row, over events within `window` steps of it. Each hop picks one of the current entity's latest `fanout` events strictly before the current time, weighted by exp(−bias × steps before the current time). Draws are keyed by the walk's root, prediction step, sample and hop, so no walk depends on other predictions or on the order of references. `period_walks` does the same inside each of the 3 previous periods, bounded by the period instead of `window`. |
| `mixed` | `mixed/changes` on, off; `mixed/trace` 0–8; `mixed/bundles` on, off | `mixed_*` is a graph of derived nodes: one per observation linked to the entity's previous one, change nodes holding feature deltas, trace nodes averaging every `trace` observations of an entity, bundle nodes averaging each step, and one node per task seen as an event. |

**Causality.** Every stage and representation reads only steps at or before the one it writes, and the stages never read labels. Truncating a truth graph after any step and transforming it gives a prefix of the transformed full graph, apart from the period-index rows of the last, unfinished period.
