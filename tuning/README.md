**Model search**

`tuning` turns a graph database into a frozen bank of truth graphs, then searches graph-transform specs and temporal graph models over it on a CUDA GPU. Models and the search read structure from the bank's pinned declaration. No code here names a node type, relation or attribute. The database contract and task file are in the [root README](../README.md), and the transforms in [graphspec/README.md](../graphspec/README.md).

**Bank**

`python -m tuning.build` reads the database once and writes each graph's truth arrays under `truth/`. Training never touches the database. Labels live in `supervision.json`, apart from the model inputs. `Bank` refuses a manifest from another bank version or whose declaration does not reproduce its fingerprint.

A trial's configuration carries its spec under `graph`. `Bank.arrays` canonicalises the spec against the model's representation groups, then materialises the variant of a graph on first use under `variants/<digest>/graph_NNNN/`. The runner materialises every graph of a trial's variant in parallel before the worker starts (`--workers`, each process holding its share of the cores) and prunes the least recently used variants to the `--variant-gb` budget, so the fit budget is spent on training; a pruned variant is renamed into `trash/` beside the store and deleted by a background thread, and folder sizes are cached by entry count, so pruning takes well under a second of the trial clock. A variant's feature matrix and query matrix are hard links to the truth's files whenever the stages leave them unchanged. The digest covers the spec without the parameters read only when packing (`time/history`, `time/context`, `neighbors/order`, `neighbors/decay`), the groups and the `graphspec` and `graph_tuning_infra` sources, so trials that differ only in those share a variant until the transform code changes. A spec with `graphs/join` set to single has one graph, index 0, merged from every truth graph before the stages run; its training windows are laid out over the merged step axis. It memory-maps the arrays and keeps the 128 most recently opened graphs' stores; a graph that packing touches holds the arrays it searches in RAM and its feature matrices on the training device, through the infra layer's resident cache, which reads them with direct file reads, holds 256 graphs and evicts the one loaded first.

**Packing**

`recipe` turns one training window into model inputs as numpy arrays and deferred gathers, with no torch work: every per-entity or per-event loop lives in a numba kernel in `gather.py` (the latest observation of an entity before a tick, neighbour sampling by recent, uniform or decay-weighted Gumbel top-k keyed by the seeded hash, the levels of a temporal-neighbour tree, interaction histories with their structural features, the events reachable from a prediction's entities, and walk identities), and edge tables are cut by binary search on their sorted columns. Feature rows are never copied on the CPU: a `Features` marker names the resident matrix, the row indices and a weight per row, and `materialise` resolves it on the device, so a masked gather, the mean of two endpoint rows or the average of an observation and a query are one device expression each. `recipe_batch` concatenates the windows of a batch in numpy; `materialise` sends every array of the batch in one bundled transfer and resolves the markers per batch rather than per window: the resident matrices of the batch's graphs are concatenated into one table per key, each window's indices are shifted by its graph's offset, and one gather serves every window. During a fit, `packing.Packer` runs the recipes in worker processes (`--workers` of them, sharing the machine's cores between their numba kernels) and writes each bundled batch into one of a fixed set of shared-memory slots (two per worker, grown to fit the largest batch seen) that the training process has registered with CUDA, so a batch moves to the device straight from the slot with no host copy and packing runs ahead of the GPU instead of alternating with it. A loader thread in the training process walks the batch schedule ahead of training and moves the resident matrices of coming graphs to the device over a side stream, throttled so that no graph still needed is evicted.

**Models**

A model is built from a model spec: a tree of parameters whose root, `view`, names the packed input the network reads, and whose other parameters switch the components that exist for that view. Every component below is read by some parameter, and every parameter is read by some component; a parameter is present in a spec only while the switch that needs it is on. `python -m tuning.check_models` runs one configuration per switch setting and records any parameter of the built network that receives no gradient; the test suite fails on one. Each view reads the base arrays plus the representation groups listed; only those groups' parameters enter its graph spec space. Each network ends in a two-layer regression head over the listed readout; every view but `mixed` also feeds the projected query to the head. Entity numbers are graph-local correspondence keys; no network learns an embedding per entity id.

| `view` | Groups | Components and their switches |
|---|---|---|
| `snapshot` | `coupling` when `temporal` is `coupling_edges` | Encoder `Linear → LayerNorm → SiLU` plus a node-kind embedding. `spatial` propagates within each observation graph, always adding a root transform of the node's own state: `none`; `mean` of neighbours; `normalised` symmetric-degree aggregation with a self loop; `relational` per-relation weight matrices with edge attributes as extra messages, one mean over all relations; `attention` single-head dot-product attention. `layers` stacks it, each with SiLU, dropout, residual and LayerNorm. `temporal` then orders the frames: `graph_sequence` pools one vector per frame, runs a query feedback round, and passes the frame sequence through `sequence` (`gru`, `lstm`, causal `attention` with `heads`); `entity_gru` a GRU cell per entity over its aligned frames; `entity_tcn` gated dilated causal convolutions of depth `temporal_layers`; `entity_attention` causal self-attention with `heads`; `graph_cell` a GRU whose gates convolve over the frame's edges, `diffusion_hops` 0 for the normalised convolution or K for K-hop forward and reverse random-walk diffusion, `temporal_layers` cells stacked; `evolving` a matrix GRU driven by top-k node summaries that rewrites the convolution weights each frame; `coupling_edges` message passing over the dated copies of each entity through coupling and period-boundary edges, `message` gated or attention, `temporal_layers` deep, closed by one mean propagation. Readout: actor, subject and context states at the query frame, with the period-context gate when `time/context` is set. |
| `mixed` | `mixed` | Message passing over the derived-node graph: `aggregation` `mean` or `attention` over edge-gated messages, `update` a GRU cell or a residual SiLU, `layers` deep. Readout: the query node's state. |
| `event_graph` | `events` | Events as nodes linked by the event DAG and by per-entity anchors, with time and relation encodings on every edge; `message` gated or attention, `layers` deep. Readout: mean of the events reachable from the query's entities, or the query node when period context adds anchors. |
| `memory` | `neighbors` when `embedding` is `attention` or `pair_terms` is on | A state per entity updated at each event tick. `update`: `mailbox_gru` builds one typed message per receiver from both endpoints' states, the event features, elapsed time, event kind, relation and status, means them and applies a GRU cell; `mutual_rnn` keeps an actor state and a subject state per entity, updates both with RNN cells at each query event and projects the actor state by log elapsed time at readout, offered only with queries; `intensity` two processes with softplus intensities over endpoint pairs, attention over the frame's edges as the neighbourhood term, then a tanh state map; `antisymmetric` integrates an antisymmetric weight matrix with `damping` by `integration_steps` Euler steps per tick, forced by event features and mean neighbour states. `embedding` reads the raw state or attends over `neighbors` sampled interaction partners. `pair_terms` replaces the two endpoint states in the head by the endpoint product, the frequency-weighted sum of common-neighbour states and each endpoint scaled by its log common-neighbour count; needs `neighbors`. |
| `projection` | `neighbors` | Seeded Gaussian projections per entity, propagated one hop order per event and decayed by `decay` per order over `hops` orders in `width` dimensions; the Gram matrix of both endpoints' projections joins a mean over `neighbors` interaction partners. |
| `sequence` | `neighbors` | Tokens are `neighbors` sampled interaction partners per endpoint with time encodings, plus `structure`: `none`, `occurrence` (two co-occurrence counts) or `overlap` (twelve pair-relative one- and two-hop features). `operator`: `mixer` token and channel MLP mixing with a mean readout; `transformer` joint attention over both endpoints' tokens folded into patches of `patch` with `heads`, root-token readout; `ssm` selective state-space blocks of state width `state` over each endpoint's tokens and over the pair's interaction times, with cross-endpoint selection. `layers` deep. |
| `tree` | `neighbors` | Recursive attention over `neighbors` sampled partners at each partner's own time, `depth` levels, time and relation encodings on tokens. |
| `walks` | `walks` | `count` causal walks of `hops` steps per endpoint, anonymised by hitting counts relative to both endpoints, encoded by a bidirectional LSTM and pooled by attention. |
| `affinity` | — | A gated linear state per actor over query categories with a recent global profile buffer of `buffer` frames. Offered for query and graph targets with queries. |
| `signal` | — | A learned directed top-`topk` adjacency over signal channels, `hops` mix-hop propagations and gated dilated convolutions, `layers` deep. Offered only with signals. |

`hidden` sets every learned width except the affinity state, whose width is the number of query categories. `dropout` is present only where a component applies it: snapshot spatial layers, causal attention in the snapshot and sequence views, temporal-edge message layers in the snapshot and event-graph views, and mixed message layers. Event and token time gaps enter the memory, projection, sequence, tree, walk, event-graph and coupling components through `cos(w · log1p(gap) + b)` with learnable `w` and `b`; period gaps enter through linear maps of `log1p(gap)`; the mixed view carries its edge gap as a linear feature; the affinity and signal views read no gaps.

The components descend from published designs and depart from them; the names above say what runs, not which paper. For the record: the graph cell follows GConvGRU and, with diffusion hops, the DCRNN cell; the evolving convolution follows EvolveGCN-H; the mailbox update follows TGN with a mean aggregator and no edge features in attention keys; the mutual RNN follows JODIE without its trajectory loss; the intensity update follows DyRep with attention over the current frame in place of its neighbourhood matrix and without the survival loss; the antisymmetric integrator follows CTAN with a mean neighbourhood term; the projections follow TPNet; the pair terms follow TNCN; the mixer follows GraphMixer with a learnable time encoding and no link features; the transformer with `occurrence` follows DyGFormer and with `overlap` HyperEvent; the selective state-space operator follows DyGMamba without patching; the tree follows TGAT with one head; the walks follow CAWN; the affinity state follows NAViS with a regression head; the signal network follows MTGNN with one gated convolution per layer. Every path trains a regression head; none of the papers' ranking, classification or point-process losses is implemented.

References: [TGAT](https://arxiv.org/abs/2002.07962), [TGN](https://arxiv.org/abs/2006.10637), [JODIE](https://arxiv.org/abs/1908.01207), [DyRep](https://arxiv.org/abs/1803.04051), [GraphMixer](https://arxiv.org/abs/2302.11636), [DyGFormer](https://arxiv.org/abs/2303.13047), [HyperEvent](https://arxiv.org/html/2507.11836v2), [TPNet](https://arxiv.org/html/2410.04013), [CAWN](https://arxiv.org/abs/2101.05974), [TNCN](https://arxiv.org/html/2406.07926v2), [DyGMamba](https://arxiv.org/html/2408.04713v3), [NAViS](https://arxiv.org/html/2510.06940v3), [CTAN](https://proceedings.mlr.press/v235/gravina24a.html) and [MTGNN](https://arxiv.org/abs/2005.11650).

**Training protocol**

- Every fit sees every training row exactly once per epoch, for 2–12 complete epochs, and scores the complete validation split. Each epoch checks the exact row ids for omissions, duplicates and validation rows.
- A training window ends at a labelled step s and covers steps s−H+1 to s, with H = `time/history`. It supervises only the labels at s. A node-target window holds at most 1,024 labels, so a step with more labels gets several windows over the same steps.
- With `time/context` c above 0, a window that starts in period p starts instead at the first step of period max(0, p−c). Each family gets its period-end readout. Replayed steps add no supervised rows.
- A batch groups windows of equal length, `batch sequences × 16 // H` of them. It holds at most 1,024 frames, 1,024 prediction rows and 1,048,576 entity-frames, where a window contributes its steps times its graph's entity count.
- The infra layer reserves the GPU memory a fit or model check may use.
- Batches hold windows of one length. They are formed within chunks of 64 graphs: each epoch shuffles the graphs, cuts them into chunks, forms every batch inside its chunk and shuffles the batches of a chunk, so the feature matrices an epoch touches are loaded about once each rather than once per batch.
- Sequence and memory states reset at window boundaries. Event models start from the last features known before the window.
- Targets are standardised with the training mean and sample standard deviation. Predictions are converted back before scoring raw MSE and R².
- The saved checkpoint is reloaded and must reproduce the first validation batch.
- Each trial has a 600-second budget measured by the runner from the moment it starts selecting the trial's parameters to the moment the variant store has been pruned after the worker exits; it covers selection, the training layout, the source snapshot, the worker's start, building the graph variant, packing, every requested epoch, scoring the holdout and the pruning. At the deadline the runner kills the worker and records a timeout; a result recorded by a worker that finished inside its own deadline but whose trial ran over the budget is set aside as `late_result.json` and the trial counts as a timeout. There is no extension. A fit that cannot complete its requested epochs in time scores nothing, so the search learns to propose epochs and structures that fit. `trial.json` records the seconds of each phase.

**Search**

| Selector | How trials are chosen |
|---|---|
| `--selector tpe` | Optuna's multivariate, grouped TPE with 10 random startup trials and 64 expected-improvement candidates. Model parameters are named `model/<parameter>`, training parameters by their names and spec parameters `graph/<parameter>`. A model parameter enters a trial only while its condition holds; a spec parameter only when the model's view reads its group and its condition holds. |
| `--selector random` | Each active model parameter, each training parameter and each active spec parameter drawn uniformly, in that order. |
| `--random-trials N` | With `tpe`, the first N trials are uniform draws recorded with selector `random`; every later proposal is TPE over all completed and failed trials so far. A resumed run keeps its N. |

For every proposal the runner rebuilds an in-memory study from the trial files; there is no external study database. A result enters the study only if it matches the frozen supervision fingerprint, covers every training row, scores the complete validation split and has a finite score. Otherwise the proposal fails. Timeouts and crashes enter TPE as failures ranked worse than every finite score. Interrupted trials are excluded.

`selection.json` records:

- the selector and its seeds; for TPE also its settings, the proposed parameters and the failure history
- the model digest, its description, the model parameters that differ from their defaults and the model space
- the spec digest, the spec parameters that differ from their defaults, and the model's spec space
- prior trial ids with result hashes
- the bank manifest hash, the supervision hash and the declaration fingerprint

Configuration and selection are saved before the worker starts, and a resumed trial reuses them. `--bounds` narrows the space both selectors draw from and the distributions TPE is told about; the bounds are stored in `plan.json` and in every `selection.json`.

A trial's configuration holds the model spec under `model`, the graph spec under `graph`, the training parameters below at the top level and `seed`. Model parameters draw from these ranges; a parameter is drawn only while its condition holds. Spec parameters and their ranges are in [graphspec/README.md](../graphspec/README.md).

| Parameter | Present when | Range |
|---|---|---|
| `view` | always | the views the bank offers |
| `hidden` | always | 16, 24, 32, 48, 64 |
| `dropout` | a component applies it | 0–0.5 |
| `snapshot/spatial`, `snapshot/layers` | view `snapshot`; layers when spatial is not `none` | none, mean, normalised, relational, attention; 1–3 |
| `snapshot/temporal` | view `snapshot` | graph_sequence, entity_gru, entity_tcn, entity_attention, graph_cell, evolving, coupling_edges |
| `snapshot/sequence` | temporal `graph_sequence` | gru, lstm, attention |
| `snapshot/temporal_layers` | temporal `entity_tcn`, `graph_cell`, `evolving`, `coupling_edges` | 1–4 |
| `snapshot/diffusion_hops` | temporal `graph_cell` | 0–4 |
| `snapshot/heads` | sequence `attention` or temporal `entity_attention` | 1, 2, 4 |
| `snapshot/message` | temporal `coupling_edges` | gated, attention |
| `mixed/aggregation`, `mixed/update`, `mixed/layers` | view `mixed` | mean, attention; gru, residual; 1–3 |
| `event_graph/message`, `event_graph/layers` | view `event_graph` | gated, attention; 1–3 |
| `memory/update` | view `memory` | mailbox_gru, mutual_rnn (with queries), intensity, antisymmetric |
| `memory/embedding`, `memory/pair_terms` | update other than `mutual_rnn` | state, attention; on, off |
| `memory/neighbors` | embedding `attention` or `pair_terms` on | 4, 8, 10, 12, 16, 20, 30, 32, 64 |
| `memory/integration_steps`, `memory/damping` | update `antisymmetric` | 1–16; 0.001–1, logarithmic |
| `projection/hops`, `projection/width`, `projection/decay`, `projection/neighbors` | view `projection` | 1–4; 16, 32, 64, 128, 256; 0.0000001–1, logarithmic; 4–64 as above |
| `sequence/operator`, `sequence/neighbors`, `sequence/layers`, `sequence/structure` | view `sequence` | mixer, transformer, ssm; 4–64 as above; 1–3; none, occurrence, overlap |
| `sequence/patch`, `sequence/heads` | operator `transformer` | 1, 2, 4, 8, 16; 1, 2, 4 |
| `sequence/state` | operator `ssm` | 4, 8, 16, 32, 64 |
| `tree/depth`, `tree/neighbors` | view `tree` | 1–2; 4, 8, 12, 16, 20 |
| `walks/count`, `walks/hops` | view `walks` | 1, 2, 4, 8, 16, 32; 1–4 |
| `affinity/buffer` | view `affinity` | 1, 2, 4, 8, 16, 32 |
| `signal/topk`, `signal/hops`, `signal/layers` | view `signal` | 1–20; 1–4; 1–3 |
| `epochs`, `clip`, `lr`, `weight_decay` | training, always | 2–12; 0.5–2; 0.0001–0.005 and 0.0000001–0.01, logarithmic |

**Run directory**

| File | Contents |
|---|---|
| `plan.json` | Bank path and manifest hash, fingerprints, seed, budget, selector, workers, variant budget and every planned trial with its model digest and description. |
| `runner.lock`, `running.json` | One runner per directory; the active worker's process id and trial. |
| `train_blocks_H.json` | The training windows for history H. |
| `trial_NNNNNN_<selector>_<model digest>_<spec digest>/` | One trial. |
| `trial_*/config.json`, `trial_*/selection.json` | The configuration, with its spec under `graph`, and how it was chosen. |
| `trial_*/worker_NN.log` | JSON events, one per line: fit entry, training progress, epochs, validation and timing logs. A failing worker ends the log with its traceback. |
| `trial_*/source/` | The `tuning`, `graphspec` and `graph_tuning_infra` sources the worker ran. |
| `trial_*/result.json`, `predictions.npz`, `checkpoint.pt` | Scores, timings, parameter count, peak GPU memory and row hashes; validation predictions; weights. |
| `trial_*/failure.json`, `timeout.json`, `fit_progress.json` | Failure records and the budget heartbeat. |
| `trial_*/trial.json`, `late_result.json` | The runner's phase timings for the whole trial, and a result that arrived after the budget. |
| `results.csv`, `results.jsonl`, `progress.md`, `coverage.json` | Every attempt with its model description, including failures; the best MSE, R², model and spec digest per view in `progress.md`; completed fits per view in `coverage.json`. |

`--trials N` ends a continuous run once N trials are recorded; a resumed run takes the limit given on resume. A file named `STOP` in the run directory stops the run after the current fit, and stops a resumed run before its first trial. The runner refuses to resume while the worker named in `running.json` is still alive.

**Verification**

`python -m tuning.check_models BANK --output checks.json [--graph N] [--spec spec.json] [--device cuda]` runs forward, backward and checkpoint reload for one model spec per setting of every structural switch the bank offers, and records any parameter that received no gradient. Each uses its default graph spec, with any of its active parameters taken from `spec.json`; a key that no view's space declares is refused. It runs on the last window of graph `N`, or of the graph with the most observations. A model that runs out of GPU memory is recorded and the check continues; the command then exits with the list of those models. Any other error stops it.
