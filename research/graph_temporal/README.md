**Graph and temporal model research**

This experiment lives outside the production advisor and production tuning services. The first pass built a frozen graph bank, implemented the architecture roster, and collected 82 small random fits. The current round adds optional past-turn-end context across every family. After an intermediate TPE phase, the continuous search returned to uniform random selection on September 10 while retaining the same targeted ranges and every accumulated result. Optuna remains available locally for a later TPE phase, without a database or integration into the production tuning system.

**Declarative graph spec**

`spec.py` is the single declaration of graph structure. Nothing else may hardcode a node kind count, a relation code, an attribute width or a design suffix. It declares:

- **Node types** (`NODES`) and the core subset (`CORE`) used by `S-core`.
- **Edge types** (`EDGES`), each with a name, a family, permitted source and target kinds, an attribute schema, a per-type default and whether it is symmetric. Canonical and reverse codes are allocated by the registry; symmetric types get one code.
- **Markers** (`MARKERS`) for the non-relation and turn-context sentinels that event and turn-replay inputs need. `RELATION_SLOTS` sizes every relation embedding.
- **Recipes** (`RECIPES`), one per design: a view, a node-type set, a set of edge families and an optional lifted layout.
- **Layout groups** (`LAYOUTS`) naming each set of alternative base recipes a model may choose among, and **modifiers** (`MODIFIERS`) declaring the family a `+suffix` adds, the base recipes it applies to and the search parameter it occupies. A modifier must cover all or none of a layout group; partial coverage is rejected at import.
- **Extraction bindings**: membership types declare the source `collection` and `subject_field` they are read from, and diplomatic types declare their source `field`, so the two regular emitters in `cache.append()` iterate the spec instead of literal tables. Irregular emitters — placement, adjacency, the war graph — stay hand-written, because extraction is domain logic rather than boilerplate.

`S-core` is a declared node-type set, not an ordinal comparison on kind indices, and adding an edge type is a spec edit rather than a change to model constructors. `Spatial` takes its relation count and its attributed types from the recipe, so the weight table and the per-type attribute projections size themselves.

**Graph selection is derived, not enumerated.** `registry.GRAPHS` declares only the graph each model family consumes — a single recipe name, or the name of a layout group when the model may choose. Everything the search needs follows from that: `registry.DESIGNS` (`spec.designs_for`), `selector.CHOICES`, `selector.STRUCTURE`, and both halves of the TPE structure space. `structure()` and `suggest_structure()` iterate `spec.modifiers_for(bases(model))` and the model's layout group instead of testing family numbers, so the propose path and the historical-import path are generated from one declaration and cannot drift apart. Models sharing a layout group share its search coordinate; a modifier is offered only to models whose layouts carry it.

**Construction recipes are declared too.** `CONSTRUCTION` holds every window, lag set, fanout and seed the builders use: the skip lags and their decision window, the all-past fanout and window, the event-DAG lags and tick window, the walk seed/samples/hops/fanout/window, the expanded-walk sample count, the turn-walk samples/hops/lookback, the turn-boundary lookback and window, and the observation-trace size. `cache.py`, `turns.py`, `walk_cache.py` and `graph_variants.py` read them instead of literals, and `graph_variants.DESIGNS` is derived from the modifier bases. The O-mixed view's node kinds and edge codes are declared as `MIXED_NODES` and `MIXED_EDGES`. All of it is inside the fingerprint, so changing a window invalidates a bank rather than silently reinterpreting it — and these are the knobs a searchable sampling recipe needs.

Adding an edge family is therefore a spec declaration, a modifier entry and an emitter. The search picks it up with no selector change: it appears in every compatible model's designs, becomes a binary TPE dimension, and round-trips through observation import. `tests/test_graph_selection.py` registers a synthetic modifier and asserts exactly that.

Edge attributes are stored per type. `spatial` keeps `(source, target, code)` triples ordered by decision step; `spatial_attribute_row` indexes into a separate `attributes_<name>` array for each attributed type, and is `-1` for types without a schema. Widening one type's schema leaves every other type's parameter shapes untouched.

The spec fingerprint is pinned into the bank manifest, the graph-variant manifest, the run plan and every trial's `selection.json`. `Bank` refuses a cache built under a different spec, and `run --resume` refuses a plan built under a different spec, so a spec edit cannot silently change what a cached array means.

The spec may declare only edges derivable from frozen source data, deterministically and causally. Model-dependent edges would break the frozen-bank property and are not permitted.

Cache format version 2 stores per-type attribute blocks and spec-allocated codes; banks built under version 1 are not readable and must be rebuilt. Model hyperparameter domains in `selector` and `registry.sample` remain indexed by model family; they describe architectures rather than graphs and are outside this spec.

**Current graph expansion and shared switches, September 10**

The current search has 110 compatible model/graph combinations across the same 30 architecture families. It samples `model` uniformly from 30 categories and then samples compatible graph edits as shared binary parameters. `turn_end` is available to every family; `colocated` is available to snapshot models M01–M11; `turn_boundary` is available to explicit temporal and event-DAG models M13–M16. M08 also selects `snapshot_graph` (core/full), and M13–M14 select `temporal_layout` (chain/skip/all-past). All three temporal layouts can enable boundary links. These switches produce the graph labels shown in the results table; the 110 labels are not a single categorical parameter.

| Graph edit | Applicable models | Added connections |
|---|---|---|
| `colocated` | M01–M11, including both M08 layouts | Both directions between characters and settlements explicitly observed at the same region in the same decision. Existing nodes and edges remain. This is a same-location shortcut, not geometric proximity or inferred reachability. |
| `turn_boundary`, observation graph | M13–M14, chain/skip/all-past | Same-entity links from its first and last observations in each of the previous three completed observed turns, within 64 decisions. First, last and coincident endpoints have distinct relation codes. |
| `turn_boundary`, event graph | M15–M16 | Shared-participant links from the first and last event timestamp groups in each of the previous three completed observed turns, within 64 decisions. Every event tied at an endpoint timestamp is retained. Existing event-DAG edges remain. |

The existing `turn_end` switch still enables each family's completed-turn endpoint readout and replay expansion. It can be combined with either applicable addition. New boundary edges are sliced to the sampled replay window; enabling `turn_boundary` alone does not enlarge that window. Short windows can therefore contain no boundary links. All additions are deterministic and causal, use no execution outcomes or labels, and add no turn-summary nodes. Neither new edge recipe removes original edges; repeated first/last endpoint references are combined into a single marked added edge.

`cache_20260909_variants` stores only the extra edge arrays, alongside the intact original, turn-end and expanded-walk banks. All 2,499 campaign files were saved, causally validated and checksummed before relaunch. Preparation and verification took 210.3 seconds; compressed campaign files total 765,276,074 bytes. Launch verifies the entire companion inventory and every checksum. Arrays are loaded on demand, and each new proposal records the companion manifest hash.

Model parameters share their coordinates across graph variants within that model. The targeted numeric domains remain range version 5, including the existing four-family restrictions. Their conditional turn-end history ranges retain separate coordinates where necessary; no range is widened or narrowed. Continuous random proposals use these same domains. Historical configurations and failure penalties remain intact for any later TPE phase, and CatBoost remains a fixed comparison only.

Verification comprises 754 distinct research checks, including causal prediction invariance, gradients reaching new relation parameters, independent campaign batching, complete frozen training coverage, historical import, score-sensitive switches and failure-sensitive TPE selection. All 110 combinations also passed a real-cache CUDA forward/backward/checkpoint check. Reports are `artifacts/graph_variants_checks_20260910.xml`, `artifacts/graph_relations_checks_20260910.xml`, `artifacts/graph_switches_checks_20260910.xml` and `artifacts/graph_variants_gpu_{none,end}_20260910.json`. Earlier sections below retain the historical design and migration details; this section defines the current parameterization.

The [live research table](D:/tw_stack/research/graph_temporal/artifacts/full_random_20260909/progress.md) tracks the continuous full-training search against the fixed greedy CatBoost baseline. The [original diagnostic report](D:/tw_stack/docs/graph_temporal_research_first_pass.md) used an incorrectly restricted 868-row training cohort. Those diagnostics remain archived and are excluded from the full-training comparison.

The [first full-training coverage report](D:/tw_stack/docs/graph_temporal_research_full_training.md) records successful fits for all 30 architecture versions and all 35 required model/graph combinations. All 35 original combinations and their 35 turn-end variants remain eligible. See the [turn-end implementation and schedule](../../docs/graph_temporal_turn_end.md). The [CatBoost audit](D:/tw_stack/docs/catboost_baseline_audit.md) verifies the fixed baseline and compares historical runs.

**Frozen graph bank**

Run from the repository root:

```powershell
.venv\Scripts\python.exe -m research.graph_temporal.cache --output research/graph_temporal/artifacts/cache_20260909 --window 2500 --workers 4
.venv\Scripts\python.exe -m research.graph_temporal.audit_cache research/graph_temporal/artifacts/cache_20260909
```

The source is one read-only PostgreSQL repeatable-read transaction. `advisor.reward_data.LiveCorpus` supplies the existing reward definition and campaign window. Its `gnn3` campaign hash split is frozen in `supervision.json`, including exact decision IDs, labels, train indices and holdout indices. The holdout is the existing validation partition, not a newly untouched test set. Labels are stored separately from graph inputs.

Each campaign has independent NumPy arrays. The completed manifest is written only after all campaigns succeed. An interrupted bank is not resumed against a different database snapshot. The audit checks every campaign, every required array, causal edge constraints, label fingerprint and exact campaign split coverage.

| View | Stored representation |
|---|---|
| S-core | Snapshot observations and typed spatial edges, restricted to faction, character, region, settlement and province types. |
| S-full | The same snapshot archive with slots, building/unit/technology/skill/item/ritual types and diplomatic relations. Canonical and reverse directions have different relation IDs. Catalogue edge attributes retain counts, levels, strength and cost where observed. |
| X-chain | Previous-observed-instance edges between dated copies of each entity. |
| X-skip | Preceding-instance lags 1, 2, 4, 8, 16, 32 and 64 within 256 decisions. |
| X-allpast | Every earlier same-entity observation within 64 decisions. These edges are materialized, not replaced with sampled neighbors. |
| O-mixed | Observation and change nodes, four-observation trace nodes, decision bundles and historical action nodes, with membership arrays and causal links. |
| D-event | State-change, relation-change and past-choice event nodes with participant chain/skip edges. Simultaneous groups have no internal DAG edges. |
| E-entity | Persistent entity IDs and a complete archive of feature updates, observed relation changes and recorded interactions. |
| B-role | Actor and subject arrays identify the two roles of each recorded interaction; role state is separate even when both refer to the same game entity. |
| H-neighbor | Per-entity temporal event indexes and per-entity observation indexes. |
| W-temporal | Eight seeded backward walks of up to four hops per endpoint/query, plus raw indexes for propagation and common-neighbor lookup. |
| N-affinity | Actor and global counts over the frozen action-type vocabulary, before the current choice. |
| V-signals | Twenty named scalar channels with observed-value masks and the current-action side input. |

The archive supports 16-, 64- and 256-decision windows. X-allpast has a 64-decision maximum. Sharing arrays does not turn these views into a common model input: each adapter reads its required representation. Neighbor sampling operates on the intact event indexes and is separate from the exact X-allpast graph.

Features use an explicit scalar allowlist, deterministic feature hashing and presence channels. Learned preprocessing is not fitted to the holdout. Entity IDs support correspondence only. Character target IDs are removed from query features; catalogue keys and known map-region identities are semantic categories. Execution validation, action results, policy scores, reward labels and run selectors are excluded. Missing values retain a missing mask.

Pre-action observations have tick `2 * step`; the recorded choice has tick `2 * step + 1`. A query may read the former and cannot read the latter. Walks strictly decrease in tick. Old observations are never filled with future state. The event archive records changes to known associations, including explicit false diplomatic statuses; absence from a partial collection is not treated as destruction.

Attack targets use recorded character/region keys. Construction queries identify actor/location slots. Research, skills and items refer to catalogue subjects. Hero-action keys supply their target. Movement uses the nearest currently located region and numeric destination coordinates; this is geographic context, not a claim about reachability. Targetless actions use a campaign-context object. Subject parsing exceptions are counted in cache metadata. In the frozen bank, 16 construction queries have no stored slot index (eight building and eight dismantling choices); they retain an actor-associated unknown-slot representation. No rows are dropped for this missing field. A parser-exception count of zero does not mean all source attributes were available.

O-mixed is a deliberately fixed cache recipe using the existing message network's node and edge conventions. Its four-observation traces and decision bundles differ from the production constructor's query-relative compression. This is an explicit graph adaptation; it does not claim identical production predictions. S-full similarly uses the permitted research schema rather than importing legacy policy/execution memory features.

Generated banks, checkpoints, predictions and logs belong under `artifacts/`, which is ignored by Git. The bank's one-time preparation cost is reported separately from each model fit.

**Implemented models**

All models predict the existing action reward. These are compact research versions with explicit task adaptations, not claims of exact published benchmark reproduction. All entity IDs are campaign-local correspondence keys. No model has a learned lookup table of campaign or character IDs.

| IDs | Version | Defining computation |
|---|---|---|
| M01–M03 | Mapgraph → GRU / LSTM / Transformer | Typed state encoder, actor/action feedback round, graph/action embedding sequence and the respective temporal operator. This is a portable mapgraph-style encoder over the permitted research schema. |
| M04 | GCN → GRU | Normalized spatial aggregation followed by aligned per-entity recurrent state. |
| M05 | GraphSAGE → GRU | Separate root and mean-neighbor transforms followed by aligned recurrence. |
| M06–M07 | RGCN → GRU / TCN | Relation-specific weight matrices, catalogue edge attributes, and recurrent or gated dilated convolutional time processing. |
| M08 | T-GCN | Graph convolution inside reset, retention and candidate gates. Both S-core and S-full are exercised. |
| M09 | DCRNN | Separate canonical forward/reverse diffusion supports inside the recurrent gates. |
| M10 | EvolveGCN-H | Observed-node top-k summaries update graph-convolution weight matrices recurrently. |
| M11 | DySAT | Structural attention followed by masked per-entity temporal attention. |
| M12 | Existing temporal-graph MPN | Reuses `TemporalRewardNet`, including graph-layer GRU updates, over the declared O-mixed cache recipe. No tuning-system entry point is called. |
| M13–M14 | Temporal-edge gated / attention | Spatial encoding, lag/relation-aware propagation on dated entity copies, then spatial propagation. Chain, skip and all-past layouts each receive a fit. |
| M15–M16 | Event-DAG gated / attention | Explicit event edges with initial feature anchors, causal lag features and participant-conditioned query readout. |
| M17 | TGAT | Recursive temporal-neighbor attention, learnable time encoding and historical feature lookup at each recursive cutoff. |
| M18 | Typed TGN | Typed messages, simultaneous-group mailboxes, GRU memory updates and temporal neighborhood embedding. Prediction precedes the current choice's memory update. |
| M19 | JODIE reward | Separate actor/object recurrent states, coupled updates and elapsed-time projection. New observations refresh side features without erasing recurrent memory. Feature/type encoders replace entity-ID embeddings. |
| M20 | DyRep reward | Separate association/interaction processes, learned intensities, current-topology attention and neighborhood-dependent memory. Reward regression replaces the original point-process objective. |
| M21 | GraphMixer | Masked token/channel mixers over interaction histories plus a separate mean-neighbor encoder. |
| M22 | DyGFormer | Paired endpoint histories, cross-history occurrence counts, patch projection and joint Transformer processing. |
| M23 | HyperEvent | Twelve pair-relative structural features spanning direct occurrence and one-/two-hop overlap, followed by a Transformer. |
| M24 | TPNet | Seeded Gaussian projections of temporal walk matrices, order-dependent exponential decay and simultaneous-group propagation. Pairwise projection inner products join temporal endpoint features in the reward head. |
| M25 | CAWN | Cached causal walks, anonymous occurrence identities relative to both query endpoints, bidirectional walk encoding and attention over walks. |
| M26 | TNCN | TGN-style memory embeddings plus explicit common-neighbor memory aggregation and endpoint-overlap terms. The initial version uses (0,1), (1,0) and (1,1) structures. |
| M27 | DyGMamba | Node-history and pair-time selective state-space blocks, input-dependent discretization and time-conditioned selection of node-history information. Uses portable PyTorch scans rather than a fused external Mamba kernel. |
| M28 | NAViS reward | Per-actor gated linear affinity state and a recent global affinity-profile buffer. Fixed action categories replace the original destination vocabulary; observed past choices supply counts, and reward regression replaces affinity ranking. |
| M29 | CTAN | Anti-symmetric state transform with damping, graph propagation and explicit numerical integration on the recorded event clock. |
| M30 | MTGNN reward | Learned directed adjacency over signal channels, mix-hop propagation, gated dilated temporal convolutions and an action-conditioned reward head. |

The local/insurance operators were inspected in `advisor/mapgraph`, `advisor/temporal_graph`, and `D:/insurance_tgcn/retention/{temporal,texas_models}.py`. Implementations are contained in this package except M12's direct reuse of the existing network class. Research references are [TGAT](https://arxiv.org/abs/2002.07962), [TGN](https://arxiv.org/abs/2006.10637), [JODIE](https://arxiv.org/abs/1908.01207), [DyRep](https://arxiv.org/abs/1803.04051), [GraphMixer](https://arxiv.org/abs/2302.11636), [DyGFormer](https://arxiv.org/abs/2303.13047), [HyperEvent](https://arxiv.org/html/2507.11836v2), [TPNet](https://arxiv.org/html/2410.04013), [CAWN](https://arxiv.org/abs/2101.05974), [TNCN](https://arxiv.org/html/2406.07926v2), [DyGMamba](https://arxiv.org/html/2408.04713v3), [NAViS](https://arxiv.org/html/2510.06940v3), [CTAN](https://proceedings.mlr.press/v235/gravina24a.html) and [MTGNN](https://arxiv.org/abs/2005.11650).

**Continuous Bayesian search with expanded configurations**

Open the [live results table](http://127.0.0.1:8766/) to see every attempt, sorted by MSE, with model, graph, search mode, timing and parameters. It refreshes every ten seconds independently of agent turns. Column headings sort the table; filters narrow it; clicking a trial opens its exact parameters and saved provenance. The fixed CatBoost baseline and unscored attempts remain visible. The viewer only reads artifacts and does not change training or the search.

Start the local viewer when it is not already running:

```powershell
.venv\Scripts\python.exe -m research.graph_temporal_dashboard.server --directory research/graph_temporal/artifacts/full_random_20260909 --port 8766
```

```powershell
.venv\Scripts\python.exe -m research.graph_temporal.check_models research/graph_temporal/artifacts/cache_20260909 --output research/graph_temporal/artifacts/model_checks.json
.venv\Scripts\python.exe -m research.graph_temporal.walk_cache research/graph_temporal/artifacts/cache_20260909
.venv\Scripts\python.exe -m research.graph_temporal.baseline --cache research/graph_temporal/artifacts/cache_20260909 --output research/graph_temporal/artifacts/greedy_catboost_20260909
.venv\Scripts\python.exe -m research.graph_temporal.run run --cache research/graph_temporal/artifacts/cache_20260909 --output research/graph_temporal/artifacts/full_random_20260909 --continuous --resume --selector tpe
.venv\Scripts\python.exe -m research.graph_temporal.run report research/graph_temporal/artifacts/full_random_20260909
```

The initial random phase shuffled a 35-entry model/design deck repeatedly. On September 9, after fit 82 finished, it switched to [Optuna's conditional multivariate TPE sampler](https://optuna.readthedocs.io/en/v4.9.0/reference/samplers/generated/optuna.samplers.TPESampler.html). `model_graph` is a categorical parameter containing every valid model/graph pair, so the optimizer selects the architecture and graph together without proposing incompatible pairs. Parameters are conditional on that pair and have separate names per pair: good settings for one architecture or graph do not overwrite the evidence for another. Multivariate TPE samples interacting settings within each conditional branch. Model allocation is now score-driven, so subsequent trials are not forced into equal coverage rounds. No family is removed, and elapsed time is not part of the objective.

Completed fits contribute raw holdout reward MSE. All compatible completed historical and new fits initialize TPE; each new completed fit is added before selecting the next trial. Recorded timeouts and crashes also contribute failure observations ranked worse than every finite completed score, as described below. Failed runs receive no prediction metrics. The fixed CatBoost score is a comparison baseline, not a neural-search observation. Successful-result import checks require the frozen supervision fingerprint, full training coverage, holdout count, history length and finite score. The sampler's ten-trial startup requirement is already satisfied by the historical fits.

Range version 4 retains the version-3 expanded domains and adds optional turn-end context with 1–3 completed turns. Version 3 expanded the parameter domains after the initial small and small-to-medium runs. Only parameters used by the chosen architecture enter its conditional search space. Every previous value remains supported, and initialization seeds remain random rather than being optimized. Comma-separated values below are categorical choices; integer ranges include every integer. Continuous ranges are uniform unless marked logarithmic.

| Parameter | Original small range | Current expanded range |
|---|---|---|
| Hidden width | 16, 24, 32 | 16, 24, 32, 48, 64 |
| Layers | 1, 2 | 1, 2, 3; TGAT retains 1, 2 recursive hops |
| Attention heads | 1, 2 | 1, 2, 4 |
| Dropout | 0, 0.1 | Continuous 0–0.5 |
| Complete epochs | 2, 3 | Integers 2–12 |
| History, decisions | 16 | Integers 1–32 |
| Neighbors | 4, 8 | 4, 8, 10, 12, 16, 20, 30, 32, 64; TGAT uses 4, 8, 12, 16, 20 |
| Learning rate, logarithmic | 0.0003–0.002 | 0.0001–0.005 |
| Weight decay, logarithmic | 0.000001–0.001 | 0.0000001–0.01 |
| Gradient clipping | Fixed at 1 | Continuous 0.5–2 |
| Diffusion/projection hops | 1, 2 | Integers 1–4 |
| Projection width | 16, 32 | 16, 32, 64, 128, 256 |
| Projection decay | 0.01, 0.05, 0.1 | Continuous 0.0000001–1, logarithmic |
| Cached walks / walk hops | 2, 4 / 2, 3 | 1, 2, 4, 8, 16, 32 / integers 1–4 |
| State-space width | 4, 8 | 4, 8, 16, 32, 64 |
| DyGFormer patch size | Fixed at 2 | 1, 2, 4, 8, 16 |
| NAViS affinity buffer | Fixed at 4 | 1, 2, 4, 8, 16, 32 |
| CTAN integration steps / damping | Fixed at 2 / 0.1 | Integers 1–16 / continuous 0.001–1, logarithmic |
| MTGNN signal neighbors | Fixed at 3 | Integers 1–20 |

TGAT's recursive depth remains at two because neighborhood expansion compounds at each hop; its maximum fanout is now 20. All attention widths divide evenly by allowed head counts. DyGFormer pads a partial final patch with masked zeros, so larger patches work with every supported neighbor count. CAWN uses a separately prepared `cache_20260909_walks32` companion cache with 32 four-hop walks. The original eight walks remain byte-identical, and the original graph bank is unchanged. Preparation checks each sampled path moves strictly into the past. Missing expanded walks fail explicitly rather than silently using fewer walks.

The [parameter review](../../docs/graph_temporal_parameter_review.md) records the published settings behind these expansions and their limitations. These are research adaptations to a reward task; timestamp units, features, operators and training schedules differ from the source benchmarks.

The new random phase draws models uniformly, then a compatible base graph and turn variant uniformly, then each active parameter independently. After 200 attempts, the saved runner switches to TPE automatically; failures count as attempts without invented scores. Completed trial files are the persistent history. For each proposal the runner rebuilds an in-memory study, imports the results once, and asks for one configuration with a reproducible proposal seed. It creates no external study or database. `selection.json` records parameters, complete distributions, range version, sampler settings/version, exact prior trial IDs, scores and result hashes. Configuration and selection are saved before the worker starts. A resumed pending trial uses its saved configuration. Earlier plans are preserved in `random_plan_before_tpe.json`, `small_plan_before_medium.json` and `plan_before_range_3.json`. Unused pre-drawn random configurations remain on disk but are excluded from the active plan. Old observations retain their original values and scores when reconstructed with the new categorical, integer and continuous distributions. Original files remain unchanged.

Every fit uses all 116,652 training decisions from 2,114 campaigns, with 2–12 complete epochs sampled per configuration. Every epoch checks the exact row IDs for omissions, duplicates and holdout contamination. Training sequences cover the same frozen campaign partition, including short tails, for every integer history length 1–32. Batches group equal-length, independent sequences, with disjoint entity IDs, edges and recurrent states. Turn variants additionally cap total replay frames per batch at 1,024, keeping a single oversized sequence intact. An original-graph batch contains up to `floor(1024 / history)` sequences, keeping at most 1,024 supervised decisions. There is no optimizer-step or training-row cap. The complete holdout contains 20,387 decisions from 385 campaigns. The 600-second deadline stops further training. If earlier completed epochs averaged strictly under 120 seconds, the current epoch may finish and the complete holdout is evaluated. Otherwise the attempt times out without a score. These exploratory schedules do not establish that any architecture has converged.

The fixed baseline uses the greedy CatBoost feature builder and existing `CB_PARAMS`: depth 8, learning rate 0.15910569655649676, up to 5,000 iterations and 50-round early stopping. It runs once, with seed 0, eight CPU threads, the frozen labels, and exactly the same campaign partition. Numeric quantization borders are fitted on training rows. Features are rebuilt from the exact frozen decision IDs, with historical chosen-action memory and battle panels strictly earlier than the query. Execution-status filters from the production row/attribution loaders are not used. This preserves the permitted CatBoost features while ensuring no decisions are selected by execution validation. The baseline is research-local and does not replace the production model. Its saved MSE appears in the same table, with each neural result's difference from it; negative differences are better.

Neural sequence/memory states reset at block boundaries. Event models initialize from permitted last-known features before the block; they do not import a trained hidden state from later observations. Each prediction sees its causal block prefix at the sampled history length. TGAT samples up to 20 recent eligible neighbors; other event-history models sample up to 64. This limit does not delete cached edges. X-allpast uses exact edges within its selected block, while the bank retains the full 64-decision version. With `turn_context=end`, blocks expand backward to replay previous completed turns and expose architecture-specific endpoint links, tokens or states. Extra context contributes no additional supervised rows. The turn-end document describes each family and the cache boundary precisely.

Each neural fit runs in its own process with a 600-second deadline and the conditional epoch-completion rule above. Data loading, parameter-dependent indexes/projections, training, full holdout scoring and checkpoint checks all fall inside that cap. Failed and timed-out fits retain their logs and failure records. Their prediction metrics remain blank while their failure outcomes inform TPE; no model family is removed. Subsequent configurations continue running. Passing `--continuous --resume` resumes the saved selector mode; it does not revert TPE to random search. A process lock prevents concurrent runners for this output directory. Before restarting after a crash, check `running.json` and the actual worker process so an orphaned worker is not overlapped. Create a file named `STOP` in the run directory to stop after the current fit. Remove it only when explicitly resuming the search.

`plan.json` records random warmup trials and subsequent TPE proposals. `train_blocks.json` retains the original history-16 layout; `train_blocks_N.json` stores each other history length. Each trial stores configuration, source snapshot, checkpoint, predictions, losses, epoch counts, timings, parameter count, GPU memory, holdout-row hash, supervision fingerprint and source fingerprint. `results.jsonl` retains completed and failed records; `results.csv` and `progress.md` include CatBoost and identify random versus TPE selection. `coverage.json` counts successful fits per model/design and requires all 70 original/turn-end combinations for complete coverage. These scores reuse a validation holdout throughout the search; they are not an untouched final test.

**Verification**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_graph_research_cache.py tests/test_graph_research_models.py tests/test_graph_research_run.py -q --basetemp=research/graph_temporal/artifacts/pytest_checks
```

Tests cover cache time boundaries, exact all-past windows, observed relation removals, feature independence from outcomes, all model/design forward/backward and checkpoint paths, future-observation/action invariance for all 30 architectures, and learning signals reaching the defining complex-model components. All 35 combinations also check that batched predictions and gradients match independent sequences. Batching retains short tails and every sequence. Real GPU checks compare batched and individual predictions across campaigns, including memory models, EvolveGCN and exact temporal edges.

The selector and runner checks verify score-sensitive model/graph selection, all-family history import, deterministic proposals, preservation of prior records, range upgrades, resume behavior, failed-trial handling and identical training-row coverage at all 32 history lengths. Model tests exercise forward/backward computation at expanded parameter limits and history lengths 1 and 32 for all 35 model/graph combinations, including partial DyGFormer patches. Walk-cache tests verify original-walk preservation, causal participant paths and invariance to future observations. The expanded suite, including turn-end variants and the random-to-TPE transition, contains 347 passing checks. All 35 turn-end combinations also passed real-cache CUDA checks.

**Epoch completion at the time limit**

At ten minutes, a fit can finish its current epoch only if its previous completed epochs averaged strictly under two minutes. It then runs full holdout evaluation and saves its score. No new epoch starts after the deadline. A fit with no previous complete epoch, or an average of two minutes or more, still times out. The same eligibility check is enforced by the process supervisor. An extended worker with no recorded progress for ten minutes is stopped.

Original sampled parameters stay unchanged in `config.json`, the results table and TPE observations. Results separately record `stopped_early`, completed `epochs`, `requested_epochs`, epoch durations and the stop reason. There is no score adjustment or parameter correction for stopping early. Full training-row coverage is checked for every completed epoch, and the full holdout is always scored.

**Targeted ranges during the random phase**

Random range version 1 applies the following overrides to new random draws. The original expanded TPE distributions stay unchanged, and all 70 model/graph choices remain eligible.

| Model | Random-only overrides |
|---|---|
| M10 EvolveGCN | Hidden width 16, 24, 32; layers 1, 2 |
| M26 TNCN | Neighbors 4, 8, 12, 16 |
| M14 Temporal-edge attention | Hidden width 16, 24; one layer |
| M29 CTAN | Integration steps 1–4 |

For the turn-end variants of these four families, history is 16–32 decisions and turn lookback is fixed at one. Larger training blocks reduce repeated replay of past turns. Other families, including DCRNN, retain their existing domains. Epochs remain 2–12; dropout, learning rate, weight decay, clipping, labels, training coverage and the timeout grace rule are unchanged.

Each new random selection records the actual reduced distributions and `random_parameter_range=targeted`, version 1. TPE still imports the original sampled parameters and unadjusted scores under the broad compatible distributions. The saved 200-attempt counter continues through the relaunch. Trial 122 was intentionally interrupted for this change, retains its original config and logs, counts as an attempted fit and has no score. The live table identifies the new policy as “Random · targeted ranges”.

**Return to TPE with the current ranges**

The user subsequently requested an immediate return to TPE, retaining the current targeted ranges. Range version 5 therefore uses the exact `random_space` domains for TPE proposals as well: the four family restrictions above, including their turn-end history/lookback restrictions, and all other existing ranges. Epochs remain 2–12. All 70 model/graph choices remain eligible. The active random trial finishes under its original budget before the switch; the original 200-attempt request and plan are archived, and the saved schedule records that the random phase ended early.

All compatible completed fits still contribute their unchanged scores and model/graph choices. Historical values outside a narrowed parameter domain cannot enter that domain's TPE distribution. Such coordinates are omitted from the narrowed sampler input, while the full original parameter mapping remains in the imported observation's `user_attrs.original_params` and the untouched result/config files. Values are never clamped or reassigned. In-range coordinates, including the original requested epochs of early-stopped fits, are imported unchanged. The dashboard marks new trials `TPE · targeted ranges`.

**Failure feedback for TPE**

At the user's request, failure policy version 1 imports historical and new `timeout.json` and `failure.json` outcomes as negative evidence. Within the temporary Optuna study, a failure has objective positive infinity so it ranks worse than every finite completed MSE. The observation uses Optuna's completed-observation state so TPE processes it, with the real failed/timeout outcome retained in its user attributes. This is a ranking penalty, not a measured MSE. Trial files, prediction metrics and dashboard failure statuses remain unchanged.

Deliberately interrupted trials, archived pending proposals and active trials are excluded. A trial that eventually has a completed result contributes that result once, instead of also importing a stale failure record. Genuine recorded crashes, including CUDA errors, and timeouts are included. The same original-parameter and narrowed-domain import rules apply to failed configurations. All model families remain eligible; this feedback influences future selection without banning a family or guaranteeing a particular configuration will never recur.

Each proposal records `failure_policy`, completed `observations`, `failed_observations`, `total_observations`, and `failure_history` with hashes of the failure record and original configuration. Successful early-stopped fits retain their actual MSE and original requested parameters, with no score adjustment. Training coverage, parameter ranges and the epoch grace rule are unchanged. Checks verify both model/graph and within-family parameter selection respond to failure evidence.

**History preparation and query computation**

M21 GraphMixer, M22 DyGFormer, M23 HyperEvent and M27 DyGMamba now prepare event histories and compute predictions only for target decisions. These models encode each query independently; their temporal operators mix that query's history tokens. Context-only queries previously repeated this work even though their predictions were discarded. The retained queries use the same causal history floor, event indexes, structural features and turn-end tokens. Stateful families still replay every context event. Batch grouping, training rows, parameter ranges and the random-to-TPE schedule are unchanged.

Deterministic checks compare target predictions and parameter gradients against computing every context query, including original and turn-end graphs, independent campaigns and empty target sets. Removing unused dropout operations can change the random sequence in training; historical checkpoints and source snapshots remain intact. A CPU benchmark over 32 cached blocks from eight training campaigns measured packing plus forward/backward speedups of 1.74× for trial 131's DyGFormer configuration and 2.81× for trial 138's HyperEvent configuration. These are sampled batch measurements, not full-fit GPU timings. Inputs, original source and measurements are saved in `artifacts/history_efficiency_20260910/`.

**Loading cached arrays on demand**

From trial 166, campaign loading opens each array only when the selected adapter requests it. Previously every campaign access outside the eight-campaign LRU opened every original array and decompressed every turn companion array, including the large walk payload for models that never read walks. The mapping still exposes every original and companion array, preserves the expanded walk override, and keeps requested arrays for the existing LRU lifetime. Cache files, model inputs, batch grouping, full training coverage and parameter ranges are unchanged.

All 70 model/graph combinations produce exactly equal packed inputs with eager and on-demand loading. The loading, model, graph, turn and walk suites pass 417 checks. A CPU packing benchmark used one block from each of 24 training campaigns, three measured repeats with alternating loader order, a warm OS file cache and the campaign LRU cleared before each batch:

| Saved configuration | Eager packing | On-demand packing | Speedup |
|---|---:|---:|---:|
| Trial 159, M30 MTGNN, V-signals + turn-end | 0.213 s | 0.108 s | 1.97× |
| Trial 162, M16 event-DAG attention, D-event + turn-end | 0.775 s | 0.693 s | 1.12× |
| Trial 171, M13 temporal-edge gated, X-allpast | 0.313 s | 0.181 s | 1.73× |

These sampled packing measurements do not establish full-fit speedups or remove all timeout causes. Real-cache packed inputs also match exactly in all three cases. The original loader, benchmark, exact blocks, source hashes and test report are retained in `artifacts/lazy_loading_20260910/`. Existing workers finish with their loaded source; new worker processes load the change automatically without interrupting the search.
