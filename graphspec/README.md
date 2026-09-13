**Graph spec core**

Domain-free machinery that reads a typed temporal truth graph from LadybugDB and derives the graph variants a model search explores. Structure comes from the database schema and a task file, both documented in the [root README](../README.md). Mechanism lives in the `graph_tuning_infra` package: files and arrays, LadybugDB access, hashing and the feature encoder.

| File | Role |
|---|---|
| `graphdb.py` | `Schema` reads node types, relations with their FROM/TO pairs, attributes, feature and signal widths and the target level back from a database. `Reader` reads one graph with Cypher anchored on the entity tables. Both go through `graph_tuning_infra.database`, which also holds the Parquet `Stage` and `copy` loaders. |
| `vocabulary.py` | `registry(schema, task)`: the base relations and every relation a spec can derive. |
| `declarations.py`, `registry.py` | `EdgeType` and `Registry`, which allocates relation codes, indexes types and fingerprints the declaration. `Registry.from_declaration` rebuilds a registry from a pinned declaration. |
| `truth.py` | `parse` turns one graph's rows into a `Graph`; `verify` checks the contract. |
| `spec.py` | The spec space: conditional parameters generated from a registry and a set of representation groups, with `draw`, `canonical` and `digest`. |
| `operators.py` | One function per graph stage. |
| `representations.py` | Model inputs built from a derived graph, and their validation. |
| `transform.py` | `derive` runs the stages and verifies the graph after each; `materialize` adds the representations. |
| `kernels.py` | The numba kernels behind the stages, the representations and `verify`. |

**Truth graph.** `truth.Graph` holds one graph as flat arrays:

- `kinds`, the node type of each entity, and `period`, the period of each step
- observations `obs_entity`, `obs_step`, `obs_x`, ordered by `(step, seq)`, and `obs_anchor`, true for an observation stage 3 keeps only because a query needs it
- facts `fact_source`, `fact_target`, `fact_code`, `fact_step`, `fact_present`, `fact_values`, ordered by step, present before absent, then `seq`
- queries `query_key`, `query_category`, `query_actor`, `query_subject`, `query_context`, `query_x`, `query_signals`
- labels `label_entity`, `label_step`, `label_row`, `label_id`

Every stage maps a `Graph` to a `Graph`, and `verify` holds after each one:

- steps in range, periods that never decrease
- canonical relation codes
- one query per step when there are queries
- one observation per entity and step
- present facts observed at their step by observations that are not anchors; queries and node labels observed at their step; absent facts observed at or before theirs

**Relation vocabulary.** `vocabulary.registry` numbers node types and base relations in database table order, then declares every relation a spec can derive. A directed relation has a canonical and a reverse code; a symmetric relation has one.

| Relation | Symmetric | Meaning |
|---|---|---|
| `<relation>` | no | A base relation, from `edge_<relation>`. |
| `project:<relation>:<side>:<A>:<B>` | yes | Entities of types A and B on one side of a relation that share a counterpart at a step, such as two units located in one region. One per relation, side and pair of types whose FROM/TO pairs share a counterpart type. |
| `complete:<A>:<B>` | yes | Pairs of completed entities of types A and B at a step. |
| `hop:2`, `hop:3` | yes | Pairs at exactly that shortest distance in the step's graph. |
| `diffusion` | no | Top-k graph-diffusion neighbours. |
| `rewire` | yes | Edges added by curvature or spectral rewiring. |
| `untyped` | no | Every base relation collapsed into one. |
| `temporal` | yes | An entity's link to its own earlier observation. |
| `boundary:first`, `boundary:last`, `boundary:both` | yes | Links from the first, last or only observation or event of an earlier period. |

The markers `non_relation` and `turn_context` take codes above every relation. The registry extras record:

- the target level
- whether queries exist
- the signal width
- the base relations
- the protected node types: the destinations of `<target>_of` for node targets, and none otherwise

`declaration()` serialises everything and `fingerprint()` hashes it. The bank manifest, the run plan and every trial's `selection.json` pin that hash.

**Spec.** A spec is a flat dictionary of graph-transform parameters. `spec.space(registry, groups)` generates the parameters for one registry and the representation groups a model reads. Each parameter is active only under its condition; `nodes/<type>/min_degree`, for example, needs `nodes/<type>/filter` set to degree or total_degree.

- `draw` walks the parameters in order and asks for a value for each active one.
- `canonical` refuses a missing active parameter, an inactive or unknown one, and an out-of-range value.
- `digest` is the SHA-256 of the sorted JSON.
- Every hashed or sampled choice in the stages and representations uses one fixed seed, `transform.SEED`, so changing one parameter never reshuffles the random choices of another.
- `materialized` drops the parameters read only when packing: `time/history`, `time/context`, `neighbors/order` and `neighbors/decay`. Its digest, with the groups and the `graphspec` and `graph_tuning_infra` sources, names a cached variant.

**Stages.** `transform.derive` applies seven stages in a fixed order. Each stage reads the graph the stage before it produced. A stage that returns its input unchanged is not verified again, and a relation whose edge filters sit at their identity values is left untouched.

1. Periods

| Parameter | Range | Effect |
|---|---|---|
| `time/periods` | truth, every | Keeps the database periods, or starts a new period every `time/period_length` steps. |
| `time/period_length` | 1–16 with queries, 2–16 without | Steps per period. Offered for every. |

2. Time

| Parameter | Range | Effect |
|---|---|---|
| `time/scheme` | window, state | Which earlier facts a snapshot holds. window: facts stated in the last w steps, or in the current period and the previous p periods. state: every earlier fact. In each scheme a fact leaves when it is removed or stated again, and an observation carries forward over the same span until the entity's next observation. Parallel facts are copied together. |
| `time/unit` | step, period | What the window counts. Offered for window when periods span more than one step: truth periods with queries, or every with a length above 1. |
| `time/window` | 1–8 | w. Offered for window counted in steps. |
| `time/past_periods` | 0–3 | p, counted in distinct periods whatever their numbers. Offered for window counted in periods. |
| `time/history` | 1–32 | Steps in each training window; read when packing. |
| `time/context` | 0–3 | Earlier periods replayed before each training window; read when packing. |

3. Nodes

Every unprotected type has its own switch and filter.

| Parameter | Range | Effect |
|---|---|---|
| `graphs/join` | separate, single | separate transforms every truth graph on its own. single merges all truth graphs of the bank into one graph before any stage: graphs follow each other in time, periods keep increasing across them, labels and splits keep their rows, and entities of connected types with the same key become one entity so campaigns meet through the catalogue they share. |
| `graphs/connect/<type>` | on, off | Whether entities of the type are unified by key when joining. Offered for single. |
| `graphs/connect_fraction` | 0.01–1, logarithmic | The share of keys that are unified, chosen by a hash of the key; the rest stay per graph. Offered for single. |
| `nodes/<type>/keep` | on, off | Removes the entities of the type and every fact that touches them. |
| `nodes/untyped` | on, off | Gives every entity one kind in the model inputs. The stages still read types, and features are untouched. Offered with more than one node type. |
| `nodes/<type>/filter` | none, degree, total_degree, core, activity, sample | Removes observations of the type's entities that fail the filter, and the facts that lose an endpoint. Offered while the type is kept. |
| `nodes/<type>/min_degree` | 1–16 | degree: an observation stays if its entity has at least that many stated incident facts among kept types at that step, a self-loop counting once. total_degree: counted over every step so far. |
| `nodes/<type>/core` | 2–16 | core: observations of every core-filtered type are removed together, repeatedly, while their incident facts to still-kept observations at that step, a self-loop counting once, number fewer than their type's threshold. Observations that failed another filter do not count. |
| `nodes/<type>/min_activity` | 2–16 | activity: an entity stays from its n-th observation, counting the copies stage 2 carried forward. |
| `nodes/<type>/sample` | 0.1–0.9 | sample: a seeded hash keeps that fraction of the type's entities for the whole graph. |

Protected types are never removed or filtered. An observation a query needs at its step is never removed; when its type is removed or it fails its type's filter, it stays as an anchor, marked in `obs_anchor`: no present fact touches it and no derivation links it, while removals of relations its entity held earlier still end at its step. Representations built on observations (coupling, events, mixed) include anchors like any other observation.

4. Derived relations

Computed at each step from the present facts that survive stage 3. Completion anchors, hops, diffusion and rewiring read the base relations that `relations/<relation>/keep` keeps, as undirected pairs. A projection reads its own relation even when that relation is removed.

| Parameter | Range | Effect |
|---|---|---|
| `derive/project:<relation>:<side>:<A>:<B>` | on, off | Adds that projection. Offered while A, B and a counterpart type their FROM/TO pairs share are kept. |
| `derive/project_cap` | 0, 2, 4, 8, 16, 32, 64 | Only each counterpart's `cap` most recently started relations of the step take part, counting parallel facts once; 0 takes all. Offered when a projection is on. |
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

Derivation stops with an error once a variant's derived facts exceed the budget the infra layer sets from the machine's memory, so a spec that completes a large type into a clique fails its trial instead of exhausting the machine.

5. Relations

| Parameter | Range | Effect |
|---|---|---|
| `relations/<relation>/keep` | on, off | Removes a base relation. Offered while both types of one of its FROM/TO pairs are kept. |
| `relations/direction` | inverse, symmetric, directed | How directed relations are read: both directions with the reverse code, both directions with one code, or the stated direction only. Symmetric relations are always read both ways. Offered while a base relation is kept. |

6. Edge filters

Every base relation and every derived relation has its own filter under `edges/<relation>/`. A base relation's filter is offered while the relation is kept; a derived relation's while a derivation can produce it: its projection; both of its completed types, with no anchor type on or an anchor type that kept relations join to both; hops of at least its distance; diffusion; or rewiring.

| Parameter | Range | Effect |
|---|---|---|
| `edges/<relation>/drop` | 0–0.9 | A seeded hash removes that fraction of entity pairs, with all their rows. |
| `edges/<relation>/persistence` | 1–4 | A present fact is kept from the p-th consecutive step it is stated; facts stated twice at one step share a position. |
| `edges/<relation>/cap` | 0–64 | At most c neighbours per entity and step, with parallel facts counted once; 0 keeps all. A directed reading caps destinations, inverse caps destinations and sources separately, and symmetric caps each entity with a pair counted once. A pair stays only if it ranks within c at every capped end. |
| `edges/<relation>/cap_order` | recent, random, largest:&lt;attribute&gt; | Which facts a cap keeps: latest run start, a seeded hash, or the largest value of one of the relation's attributes. Offered when the cap is on. |

Relations read symmetrically count their two directions as one pair in every edge filter.

Whenever a stage removes present facts, it also removes absent rows left without any earlier kept present row. The event history therefore never removes a relation the variant never stated.

7. Relation merging

| Parameter | Range | Effect |
|---|---|---|
| `relations/types` | typed, untyped | untyped replaces every base relation with `untyped` and drops attributes. Its absent row stays only when no underlying relation between the pair is still present. Offered while a base relation is kept. |
| `relations/parallel` | keep, merge | merge keeps the last of identical facts stated at one step. Offered while a base relation is kept. |

**Representations.** `representations.build` always produces the base arrays and adds the groups a model reads.

Base arrays:

- snapshots: `x`, `entity`, `step`, `turn` and `snapshot_offsets` per observation; `observation_ids` and `observation_offsets` per entity; `entity_kind`, all zero when nodes are untyped; relation edges `spatial` as source observation, target observation and code, with `spatial_step`, `spatial_attribute_row` and `attributes_<relation>`
- predictions: `actor`, `subject`, `query_context`, `query_step`, `row_index`, `decision`, `signals`; a node target is its own actor and subject
- queries: `query` and `decision_actor`
- `affinity` and `global_affinity`: counts of earlier query categories, per actor and overall
- `events`: one row per observation change, relation change or query. Observations and facts of step s sit at tick 2s and the query of step s at tick 2s+1.
- periods: `period`, and the index `turn_starts`, `turn_ends`, `turn_numbers`, `turn_membership`, `turn_observations`, `turn_actor_decision`, `turn_action_category`

| Group | Parameters | Arrays |
|---|---|---|
| `coupling` | `coupling/scheme` ordinal, lags, window, all; `coupling/unit` observation, step, for ordinal and lags; `coupling/max_lag` 2, 4, 8, 16, 32, 64, for lags; `coupling/lag_window` 1–512, for lags by observation; `coupling/window` 2–128 and `coupling/cap` 1–128, for window; `coupling/periods` 0–3; `coupling/period_window` 2–128, for periods above 0 | `coupling` links an observation to earlier observations of its entity. ordinal takes the previous one. lags takes those 1, 2, 4, … up to `max_lag` back, within `lag_window` steps. window takes every earlier one less than `window` steps back, at most the latest `cap`. all takes every earlier one. With unit step, ordinal and lags count steps and need an observation exactly that far back. `coupling_boundaries` links from the entity's first and last observations in each of the previous `periods` periods, less than `period_window` steps back. |
| `events` | `events/kinds` any subset of observation, relation and, with queries, query; `events/link` next, lags, window; `events/max_lag` 2, 4, 8, 16, for lags; `events/window` 1–1024; `events/cap` 0–64; `events/periods` 0–3; `events/period_window` 2–128, for periods above 0 | `event_dag` links each member event, once per earlier event, from its participants' earlier tick groups within `window` ticks. next takes the previous group, lags the groups 1, 2, 4, … up to `max_lag` back, and window every earlier group. With `cap` above 0, only the last `cap` events of each group link. `event_member` marks the chosen kinds. `event_boundaries` links from the first and last tick groups of each participant's previous `periods` periods, less than `period_window` steps back. |
| `neighbors` | `neighbors/kinds` relation and, with queries, query or both; `neighbors/removals` on, off, when relation events are included; `neighbors/order` recent, uniform, decay; `neighbors/decay` 0.001–1, log, for decay | `neighbor_ids` and `neighbor_offsets` index each entity's relation events, with or without removals, and query events. When packing, a model takes its neighbour count from before each cutoff: the latest, a uniform sample, or a sample weighted by exp(−decay × elapsed steps), drawn by Gumbel top-k with keys hashed from the entity, the cutoff and the candidate event. Candidates start at the packed window's first step. |
| `walks` | `neighbors/kinds`, `neighbors/removals`; `walks/bias` 0–1; `walks/window` 1–512; `walks/fanout` 1–128 | Builds the neighbour index as above. `walks` holds 32 walks of 4 hops from the actor and the subject of each prediction, over events within `window` steps of it. Each hop picks one of the current entity's latest `fanout` events strictly before the current time, weighted by exp(−bias × steps before the current time). Draws are keyed by the walk's root, prediction step, side, sample and hop, so no walk depends on other predictions. `turn_walks` does the same inside each of the 3 previous periods, bounded by the period instead of `window`. |
| `mixed` | `mixed/changes` on, off; `mixed/trace` 0–8; `mixed/bundles` on, off | `mixed_*` is a graph of derived nodes: one per observation linked to the entity's previous one, change nodes holding feature deltas, trace nodes averaging every `trace` observations of an entity, bundle nodes averaging each step, and one action node per query. |

**Causality.** Every stage and representation reads only steps at or before the one it writes, and the stages never read labels. Truncating a truth graph after any step and transforming it gives a prefix of the transformed full graph, apart from the period-index rows of the last, unfinished period.
