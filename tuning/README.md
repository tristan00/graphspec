**Model search**

`tuning` turns staged graph tables or a graph database into a frozen bank of truth graphs, then searches graph-transform specs and temporal graph models over it on a CUDA GPU. Models and the search read structure from the bank's pinned declaration. No code here names a node type, relation or attribute. The database contract and task file are in the [root README](../README.md), and the transforms in [graphspec/README.md](../graphspec/README.md).

Commands require an absolute `GRAPHSPEC_HOME` before Python starts. Use the checkout launcher with mandatory `-Home`, for example `./scripts/graphspec.ps1 -Home D:/graphspec-runtime search run --bank D:/graphspec-runtime/banks/churn --output D:/graphspec-runtime/runs/churn-tuning --continuous --selector botorch --startup-trials 100 --random-params model/view`. Outputs stay outside code directories and, on Windows, on the configured runtime volume. Caches and temporary files live under that home. No profile settings or default home are consulted; see [storage](../docs/loader.md).

**Bank**

`python -m tuning.build` reads its explicit source once and writes each graph's truth arrays under `truth/`. Training never touches the database. Labels live in `supervision.json`, apart from the model inputs. `Bank` refuses a manifest from another bank version or whose declaration does not reproduce its fingerprint.

`Bank.arrays` materialises graph variants under `cache/banks/<bank-digest>/variants/<digest>/graph_NNNN/`. The digest includes the transform source, representation groups, and graph parameters, excluding parameters used only during packing. The runner prepares variants with `--workers` and prunes them to `--variant-gb` between trials. Preparation counts against the trial deadline.

`graphs/join=single` joins groups of `graphs/join_size` truth graphs, each onto one step axis, and builds each group campaign by campaign with the segment engine (see the graphspec README), so memory stays near one campaign whatever the group size. Groups build in parallel across `--workers`; when a split has fewer groups than workers, the workers are spread over its campaigns. Windows of a joined group use a window-local entity index: the entities the window references, renumbered, instead of every entity of the group, and the batch budget counts each window's distinct observed entities. A batch draws its windows from groups covering about 64 campaigns, as separate graphs do. Joined arrays up to 4 GiB are held on the training device, and resident joined graphs are evicted least recently used beyond 12 GiB. Large joined arrays use temporary memory-mapped files and bounded copies on the runtime volume. Turn history uses sorted observation and actor indexes. The array store retains 128 graph stores; the resident cache retains 192 graphs. Arrays over 64 MiB remain mapped, and packing transfers only requested feature rows to the training device. Observation features are stored once: a variant holds `x_ref`, one row number per observation, and the feature table `x_table`; a separate graph's table is a file copy of the truth graph's `obs_x` when no stage changed it. A batch selects rows from each graph's resident arrays and concatenates only what it selected.

**Packing**

`data.py` owns bank storage and variant identity, `inputs.py` builds model inputs from graph windows, and `tensors.py` resolves deferred gathers onto the training device. `batching.py` combines independent windows and `packing.py` owns worker processes and their transfer buffers.

`recipe` prepares one window as NumPy arrays and deferred feature gathers. Numba kernels implement graph traversal and sampling; sorted indexes bound row searches. `recipe_batch` combines equal-length independent windows, and `materialise` resolves their gathers together while preserving row order and duplicates. Views whose rows do not interact (tree, walks, sequence, mixed, event graph) narrow a window's rows to the span of its supervised rows; memory and projection compute neighbour histories for supervised rows only. For the snapshot view, except with coupling edges, a batch builds each graph's overlapping frames once: the input and spatial layers run on those frames and the result is expanded to the windows, and per-window edges are built only for the graph-cell and evolving modes, which read them per frame.

`packing.Packer` prepares batches in worker processes using two shared-memory slots per worker. Worker processes start with their numeric libraries limited to the worker's thread share, which keeps an idle worker near 1 GB. CUDA receives each packed batch from a registered slot. A prefetch thread loads upcoming resident arrays within the graph-cache window; large mapped tables gather rows on demand. Small feature tables remain on the training device.

**Models**

A model is built from a model spec: a tree of parameters whose root, `view`, names the packed input the network reads, and whose other parameters switch the components that exist for that view. Every component below is read by some parameter, and every parameter is read by some component; a parameter is present in a spec only while the switch that needs it is on. `python -m tuning.check_models` runs one configuration per switch setting and records any parameter of the built network that receives no gradient; the test suite fails on one. Each view reads the base arrays plus the representation groups listed; only those groups' parameters enter its graph spec space. Each network ends in a two-layer regression head over the listed readout; every view but `mixed` also feeds the projected query to the head. Entity numbers are graph-local correspondence keys; no network learns an embedding per entity id.

| `view` | Groups | Components and their switches |
|---|---|---|
| `snapshot` | `coupling` when `temporal` is `coupling_edges` | Encoder `Linear → LayerNorm → SiLU` plus a node-kind embedding. `spatial` propagates within each observation graph, always adding a root transform of the node's own state: `none`; `mean` of neighbours; `normalised` symmetric-degree aggregation with a self loop; `relational` per-relation weight matrices with edge attributes as extra messages, one mean over all relations; `attention` single-head dot-product attention; `chebyshev` a Chebyshev polynomial filter of order `chebyshev_order` in the symmetric-normalised Laplacian of the frame, one weight matrix per order; `relational_attention` dot-product attention whose keys and messages carry a learned relation embedding. `layers` stacks it, each with SiLU, dropout, residual and LayerNorm. `temporal` then orders the frames: `graph_sequence` pools one vector per frame, runs a query feedback round, and passes the frame sequence through `sequence` (`gru`, `lstm`, causal `attention` with `heads`); `entity_gru` a GRU cell per entity over its aligned frames; `entity_tcn` gated dilated causal convolutions of depth `temporal_layers`; `entity_attention` causal self-attention with `heads`; `graph_cell` a GRU whose gates convolve over the frame's edges, `diffusion_hops` 0 for the normalised convolution or K for K-hop forward and reverse random-walk diffusion, `temporal_layers` cells stacked; `evolving` a matrix GRU driven by top-k node summaries that rewrites the convolution weights each frame; `coupling_edges` message passing over the dated copies of each entity through coupling and period-boundary edges, `message` gated or attention, `temporal_layers` deep, closed by one mean propagation. Readout: actor, subject and context states at the query frame, with the period-context gate when `time/context` is set. |
| `mixed` | `mixed` | Message passing over the derived-node graph: `aggregation` `mean` or `attention` over edge-gated messages, `update` a GRU cell or a residual SiLU, `layers` deep. Readout: the query node's state. |
| `event_graph` | `events` | Events as nodes linked by the event DAG and by per-entity anchors, with time and relation encodings on every edge; `message` gated or attention, `layers` deep. Readout: mean of the events reachable from the query's entities, or the query node when period context adds anchors. |
| `memory` | `neighbors` when `embedding` is `attention` or `pair_terms` is on | A state per entity updated at each event tick. `update`: `mailbox_gru` builds one typed message per receiver from both endpoints' states, the event features, elapsed time, event kind, relation and status, means them and applies a GRU cell; `mutual_rnn` keeps an actor state and a subject state per entity, updates both with RNN cells at each query event and projects the actor state by log elapsed time at readout, offered only with queries; `intensity` two processes with softplus intensities over endpoint pairs, attention over the frame's edges as the neighbourhood term, then a tanh state map; `antisymmetric` integrates an antisymmetric weight matrix with `damping` by `integration_steps` Euler steps per tick, forced by event features and mean neighbour states. `embedding` reads the raw state or attends over `neighbors` sampled interaction partners. `pair_terms` replaces the two endpoint states in the head by the endpoint product, the frequency-weighted sum of common-neighbour states and each endpoint scaled by its log common-neighbour count; needs `neighbors`. |
| `projection` | `neighbors` | Seeded Gaussian projections per entity, propagated one hop order per event and decayed by `decay` per order over `hops` orders in `width` dimensions; the Gram matrix of both endpoints' projections joins a mean over `neighbors` interaction partners. |
| `sequence` | `neighbors` | Tokens are `neighbors` sampled interaction partners per endpoint with time encodings, plus `structure`: `none`, `occurrence` (two co-occurrence counts) or `overlap` (twelve pair-relative one- and two-hop features). `operator`: `mixer` token and channel MLP mixing with a mean readout; `transformer` joint attention over both endpoints' tokens folded into patches of `patch` with `heads`, root-token readout; `ssm` selective state-space blocks of state width `state` over each endpoint's tokens and over the pair's interaction times, with cross-endpoint selection. `layers` deep. |
| `tree` | `neighbors` | Recursive attention over `neighbors` sampled partners at each partner's own time, `depth` levels, time and relation encodings on tokens. |
| `walks` | `walks` | `count` causal walks of `hops` steps per endpoint, anonymised by hitting counts relative to both endpoints, encoded by a bidirectional LSTM and pooled by attention. |
| `affinity` | — | A gated linear state per actor over query categories with a recent global profile buffer of `buffer` frames. Offered for query and graph targets with queries. |
| `signal` | — | A learned directed top-`topk` adjacency over signal channels, `hops` mix-hop propagations and gated dilated convolutions, `layers` deep. Offered for query and graph targets with signals. |

`hidden` sets every learned width except the affinity state, whose width is the number of query categories. `dropout` is present only where a component applies it: snapshot spatial layers, causal attention in the snapshot and sequence views, temporal-edge message layers in the snapshot and event-graph views, and mixed message layers. Event and token time gaps enter the memory, projection, sequence, tree, walk, event-graph and coupling components through `cos(w · log1p(gap) + b)` with learnable `w` and `b`; period gaps enter through linear maps of `log1p(gap)`; the mixed view carries its edge gap as a linear feature; the affinity and signal views read no gaps.

The components descend from published designs and depart from them; the names above say what runs, not which paper. For the record: the graph cell follows GConvGRU and, with diffusion hops, the DCRNN cell; the evolving convolution follows EvolveGCN-H; the mailbox update follows TGN with a mean aggregator and no edge features in attention keys; the mutual RNN follows JODIE without its trajectory loss; the intensity update follows DyRep with attention over the current frame in place of its neighbourhood matrix and without the survival loss; the antisymmetric integrator follows CTAN with a mean neighbourhood term; the projections follow TPNet; the pair terms follow TNCN; the mixer follows GraphMixer with a learnable time encoding and no link features; the transformer with `occurrence` follows DyGFormer and with `overlap` HyperEvent; the selective state-space operator follows DyGMamba without patching; the tree follows TGAT with one head; the walks follow CAWN; the affinity state follows NAViS with a regression head; the signal network follows MTGNN with one gated convolution per layer. Every path trains a regression head; none of the papers' ranking, classification or point-process losses is implemented.

References: [TGAT](https://arxiv.org/abs/2002.07962), [TGN](https://arxiv.org/abs/2006.10637), [JODIE](https://arxiv.org/abs/1908.01207), [DyRep](https://arxiv.org/abs/1803.04051), [GraphMixer](https://arxiv.org/abs/2302.11636), [DyGFormer](https://arxiv.org/abs/2303.13047), [HyperEvent](https://arxiv.org/html/2507.11836v2), [TPNet](https://arxiv.org/html/2410.04013), [CAWN](https://arxiv.org/abs/2101.05974), [TNCN](https://arxiv.org/html/2406.07926v2), [DyGMamba](https://arxiv.org/html/2408.04713v3), [NAViS](https://arxiv.org/html/2510.06940v3), [CTAN](https://proceedings.mlr.press/v235/gravina24a.html) and [MTGNN](https://arxiv.org/abs/2005.11650).

**Training protocol**

- The optimizer is AdamW, fused on CUDA.
- Every fit sees every training row exactly once per epoch, for every requested epoch, and scores the complete validation split. Each epoch checks the exact row ids for omissions, duplicates and validation rows.
- A training window ends at a labelled step s and covers steps s−H+1 to s, with H = `time/history`. It supervises only the labels at s. A node-target window holds at most 1,024 labels, so a step with more labels gets several windows over the same steps.
- With `time/context` c above 0, a window that starts in period p starts instead at the first step of period max(0, p−c). Each family gets its period-end readout. Replayed steps add no supervised rows.
- A batch groups windows of equal length, `batch sequences × 16 // H` of them. It holds at most 1,024 frames, 1,024 prediction rows, 1,048,576 entity-frames and 4,194,304 facts, where a window contributes its steps times its graph's entity count and times its graph's facts per step from the manifest.
- The infra layer reserves the GPU memory a fit or model check may use.
- Batches hold windows of one length. They are formed within chunks of 64 graphs: each epoch shuffles the graphs, cuts them into chunks, forms every batch inside its chunk and shuffles the batches of a chunk, so the feature matrices an epoch touches are loaded about once each rather than once per batch.
- Sequence and memory states reset at window boundaries. Event models start from the last features known before the window.
- Targets are standardised with the training mean and sample standard deviation. Predictions are converted back before scoring raw MSE and R².
- The saved checkpoint is reloaded and must reproduce the first validation batch.
- The configured `--seconds` budget covers selection, layout, source capture, variant building, packing, training, and evaluation. At the deadline the runner records a timeout and waits for every owned worker to exit before continuing. Pruning runs after cleanup, outside the trial budget. Late outputs are retained with `late_` names; incomplete trials receive no prediction score. `trial.json` records phase timings.

**Search**

| Selector | How trials are chosen |
|---|---|
| `--selector forest` | SMAC-style selection: a random forest fitted to the log of completed scores predicts each candidate's MSE, the spread across its trees stands in for uncertainty, and a second forest fitted to all completed, failed and timed-out trials predicts completion. Log expected improvement over the best observed MSE, computed in log space, plus three times the log completion probability ranks 10000 uniformly drawn valid configurations. Parameters selected by `--random-params` are drawn first and held fixed. |
| `--selector botorch` | The default for new runs. BoTorch fits a MAP-SAAS Gaussian process to completed scores and a separate feasibility model to all completed, failed and timed-out trials. Both models fit on the GPU, and log expected improvement ranks 10000 uniformly drawn valid configurations to pick one. Parameters selected by `--random-params` are drawn first and held fixed in every candidate evaluated by the model. |
| `--objective` | For path banks, `series=weight` pairs that blend the per-series validation losses into the one score trials are ranked by, for example `--objective gdp=1 population=1 sol=1`. Each series loss is its query-weighted validation MSE divided by the series' training variance. A new run weighs every series 1; a continued run keeps the saved weights unless new ones are given, and since every result stores each series' loss, reweighting re-ranks finished trials without refitting. Other banks rank by raw validation MSE and take no weights. |
| `--mix` | Percent of trials after the startup phase handed to each selector, as `name=share` pairs summing to 100, for example `--mix random=80 forest=10 tpe=10`; a new run defaults to 100 for its `--selector`, otherwise the saved mix is kept. |
| `--selector tpe` | Optuna's multivariate, grouped TPE with 10 random startup trials, 64 expected-improvement candidates, and a good set of the best 100 trials. Model parameters are named `model/<parameter>`, training parameters by their names and spec parameters `graph/<parameter>`. A model parameter enters a trial only while its condition holds; a spec parameter only when the model's view reads its group and its condition holds. |
| `--selector random` | Each active model parameter, each training parameter and each active spec parameter drawn uniformly, in that order. |
| `--startup-trials N` | Minimum number of finished trials before BoTorch starts; defaults to 100. Completed, failed and timed-out trials count regardless of their original selector. Below this threshold new trials are random. Existing history is retained. `--random-trials` is an alias; plans retain the `random_trials` field. Legacy TPE uses this number as its initial trial-index threshold. |
| `--random-params model/view ...` | Fully qualified parameter names or wildcard patterns to keep random after BoTorch starts. Model names use `model/`, graph names use `graph/`, and training names are unprefixed. Draws obey the proposal bounds and remain fixed whenever active. Omission preserves a continued run's setting; supplying the flag without names clears it. |

For every proposal the runner rebuilds the optimizer from the trial files; there is no external study database. A completed result must match the frozen supervision fingerprint, cover every training row, score the complete validation split and have a finite score. Otherwise loading the result fails. BoTorch learns from the original recorded configurations, including history outside the current proposal bounds. Bounds constrain new proposals, not historical training inputs. Categorical values use one-hot encoding; numeric inputs use their declared ranges and logarithmic scales, with activity indicators for conditional values. Failed trials train feasibility without invented prediction scores. TPE retains its existing failure ranking and bound filtering. Interrupted trials are excluded from both learning and BoTorch's startup count.

`selection.json` records:

- the selector, seeds, settings, proposed parameters and failure history; BoTorch also records preset random values and every trial used for learning
- the model digest, its description, the model parameters that differ from their defaults and the model space
- the spec digest, the spec parameters that differ from their defaults, and the model's spec space
- prior trial ids with result hashes
- the bank manifest hash, the supervision hash and the declaration fingerprint

Configuration and selection are saved before the worker starts, and a continued trial reuses them. `--bounds` narrows the space new proposals draw from; the bounds are stored in `plan.json` and in every `selection.json`.

Each new trial saves its time limit in seconds as `config.json`'s `max_seconds`. The runner enforces that saved limit. Changing `--seconds` on continuation changes the limit for new proposals; omitting it keeps the run's current setting. The dashboard and result exports expose older trials' recorded budgets under the same parameter without rewriting their original configurations.

To draw epochs and runtime randomly before BoTorch chooses the remaining parameters, use `--random-params model/view epochs max_seconds` with bounds `{"epochs":{"low":1,"high":5},"max_seconds":{"low":600,"high":1200}}`. Both ranges include every integer between their endpoints. The `max_seconds` bound takes precedence over `--seconds` for new proposals. BoTorch receives both preset values in every candidate and learns from historical runtime limits recorded with completed and failed trials. Missing historical limits remain unknown; the current plan's limit is never substituted for them.

A trial's configuration holds the model spec under `model`, the graph spec under `graph`, the training parameters below at the top level and `seed`. Model parameters draw from these ranges; a parameter is drawn only while its condition holds. Spec parameters and their ranges are in [graphspec/README.md](../graphspec/README.md).

| Parameter | Present when | Range |
|---|---|---|
| `view` | always | the views the bank offers |
| `hidden` | always | Every integer from 1 through 128; uniform integer sampling for random trials and an integer distribution for TPE. |
| `dropout` | a component applies it | 0–0.5 |
| `time/encoding` | a view encodes time gaps: `event_graph`, `memory`, `projection`, `sequence`, `tree`, `walks`, and `snapshot` with `coupling_edges` | log_cosine, cosine, log_linear |
| `keys/embedding`, `keys/table` | a view embeds entities: every view but `affinity` and `signal`; table when on | off, on; 4096–65536 |
| `snapshot/spatial`, `snapshot/layers` | view `snapshot`; layers when spatial is not `none` | none, mean, normalised, relational, attention, chebyshev, relational_attention; 1–3 |
| `snapshot/chebyshev_order` | spatial `chebyshev` | 1–4 |
| `snapshot/temporal` | view `snapshot` | graph_sequence, entity_gru, entity_tcn, entity_attention, graph_cell, evolving, coupling_edges |
| `snapshot/sequence` | temporal `graph_sequence` | gru, lstm, attention |
| `snapshot/temporal_layers` | temporal `entity_tcn`, `graph_cell`, `evolving`, `coupling_edges` | 1–4 |
| `snapshot/diffusion_hops` | temporal `graph_cell` | 0–4 |
| `snapshot/heads` | sequence `attention` or temporal `entity_attention` | 1–4 |
| `snapshot/message` | temporal `coupling_edges` | gated, attention |
| `mixed/aggregation`, `mixed/update`, `mixed/layers` | view `mixed` | mean, attention; gru, residual; 1–3 |
| `event_graph/message`, `event_graph/layers` | view `event_graph` | gated, attention; 1–3 |
| `memory/update` | view `memory` | mailbox_gru, mutual_rnn (with queries), intensity, antisymmetric |
| `memory/embedding`, `memory/pair_terms` | update other than `mutual_rnn` | state, attention; on, off |
| `memory/neighbors` | embedding `attention` or `pair_terms` on | 4–64 |
| `memory/integration_steps`, `memory/damping` | update `antisymmetric` | 1–16; 0.001–1, logarithmic |
| `projection/hops`, `projection/width`, `projection/decay`, `projection/neighbors` | view `projection` | 1–4; 16–256; 0.0000001–1, logarithmic; 4–64 as above |
| `sequence/operator`, `sequence/neighbors`, `sequence/layers`, `sequence/structure` | view `sequence` | mixer, transformer, ssm; 4–64 as above; 1–3; none, occurrence, overlap |
| `sequence/patch`, `sequence/heads` | operator `transformer` | 1–16; 1–4 |
| `sequence/state` | operator `ssm` | 4–64 |
| `tree/depth`, `tree/neighbors` | view `tree` | 1–2; 4–20 |
| `walks/count`, `walks/hops` | view `walks` | 1–32; 1–4 |
| `affinity/buffer` | view `affinity` | 1–32 |
| `signal/topk`, `signal/hops`, `signal/layers` | view `signal` | 1–20; 1–4; 1–3 |
| `epochs`, `clip`, `lr`, `weight_decay` | training, always | 2–12; 0.5–2; 0.0001–0.005 and 0.0000001–0.01, logarithmic |

**Run directory**

The runner owns one trial at a time. Windows Job Objects and Linux process groups contain its descendants; uncertain cleanup stops the runner. `tuning.records` validates trial identity, complete epoch and row coverage, finite scores, and outcome files. Completed, failed, timed-out, and interrupted attempts stay distinct; a worker that exited without an outcome is interrupted. Older plan versions require explicit migration.

| File | Contents |
|---|---|
| `plan.json` | Bank path and manifest hash, fingerprints, seed, budget, selector, workers, variant budget and every planned trial with its model digest and description. |
| `runner.lock` | Per-run writer lock in the external run directory. Windows heavy-job singletons use user-SID named mutexes without lock files. |
| `runner.json`, `running.json` | Runner lifecycle and owner identity; active worker identity and trial. |
| `train_blocks_H.json` | The training windows for history H. |
| `trial_NNNNNN_<selector>_<model digest>_<spec digest>/` | One trial. |
| `trial_*/config.json`, `trial_*/selection.json` | The configuration, with its spec under `graph`, and how it was chosen. |
| `trial_*/worker_NN.log` | JSON events, one per line: fit entry, training progress, epochs, validation and timing logs. A failing worker ends the log with its traceback. |
| `trial_*/source/` | The `tuning`, `graphspec` and `graphspec.infra` sources the worker ran. |
| `trial_*/result.json`, `predictions.npz`, `checkpoint.pt` | Scores, timings, parameter count, peak GPU memory and row hashes; validation predictions; weights. |
| `trial_*/failure.json`, `timeout.json`, `fit_progress.json` | Failure records and the budget heartbeat. |
| `trial_*/trial.json`, `late_result.json` | The runner's phase timings for the whole trial, and a result that arrived after the budget. |
| `results.csv`, `results.jsonl`, `progress.md`, `coverage.json` | Every attempt with its model description, including failures; the best MSE, R², model and spec digest per view in `progress.md`; completed fits per view in `coverage.json`. |

`--trials N` ends a continuous run once N trials are recorded; a continued run takes the limit it is given. A file named `STOP` in the run directory stops the run after the current fit, and stops a continued run before its first trial. The runner refuses to continue while the worker named in `running.json` is still alive.

**Verification**

`python -m tuning.check_models BANK --output checks.json [--graph N] [--spec spec.json] [--device cuda]` runs forward, backward and checkpoint reload for one model spec per setting of every structural switch the bank offers, and records any parameter that received no gradient. Each uses its default graph spec, with any of its active parameters taken from `spec.json`; a key that no view's space declares is refused. It runs on the last window of graph `N`, or of the graph with the most observations. A model that runs out of GPU memory is recorded and the check continues; the command then exits with the list of those models. Any other error stops it.

Two switches apply across views. `time/encoding` picks the time-gap encoding every temporal component uses: `log_cosine` is `cos(log1p(gap) w + b)` with learnable `w`, `cosine` the same on the raw gap, `log_linear` a linear map of `log1p(gap)`. `keys/embedding` adds a learned vector per entity key, hashed into a table of `keys/table` rows, to every entity encoder of the view: observation nodes, derived nodes, event endpoints, memory states, tree, history and walk nodes, and the actor and subject roots. Only node types whose keys recur across graphs receive it: a bank marks a type shared when at least half of its distinct keys appear in two or more graphs, and other types get a zero vector, so identifiers that are unique to one graph, such as characters, never carry an embedding that could memorise the graph they came from. The table is a parameter trained by the training rows only, so holdout graphs see vectors learned from training graphs that contain the same key.
